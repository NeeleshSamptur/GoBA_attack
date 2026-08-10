"""
experiments/robot/libero/run_dof_timeseries_metrics.py

TIME-SERIES COMPARISON OF PER-DoF ATTENTION MAPS: does DoF d's action-query
map over the 256 image patches change from one control step to the next, and
does that drift differ between clean/poison/decoy rollouts?

Reads an existing trajectory_dof_attention-style npz (maps: n_episodes x
T_steps x 7 DoF x 256 patches, produced by run_trajectory_dof_attention.py or
its BadVLA/AttackVLA/clean-model counterparts) -- no model re-run needed.
Takes the first STEPS control steps of every episode and, per DoF, compares
consecutive-step maps (t -> t+1) with six metrics:

  cosine   - flattened cosine similarity (scale-invariant, shift-blind)
  pearson  - flattened Pearson correlation
  spearman - flattened Spearman rank correlation
  js       - Jensen-Shannon divergence, base-2 (maps already sum to 1)
  kl       - KL(map_t || map_t+1), maps as distributions
  emd      - 2D Wasserstein distance on the 16x16 patch grid (POT exact OT,
             ground cost = Euclidean distance between patch (row,col))
  ssim     - structural similarity on the 16x16 reshaped maps (skimage)

Outputs: attn_maps/dof_timeseries_metrics_<npz_stem>.npz (per-episode,
per-DoF, per-transition metric arrays) + a printed summary: per-metric
overall stats, per-DoF breakdown, per-transition (time) trend, and
per-role means with poison-vs-clean AUROC (mirrors the convention used by
every other script in this pipeline).
"""
import argparse
import os

import numpy as np
import ot
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as kl_entropy
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
N_DOF = 7
EPS = 1e-12


def normalize(m):
    s = m.sum()
    return m / s if s > 0 else np.full_like(m, 1.0 / m.size)


def grid_coords(side):
    rr, cc = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    return np.stack([rr.ravel(), cc.ravel()], axis=1).astype(np.float64)


def compute_pair_metrics(a, b, cost, side):
    a, b = normalize(a.astype(np.float64)), normalize(b.astype(np.float64))
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))
    pear = float(pearsonr(a, b)[0])
    spear = float(spearmanr(a, b)[0])
    js = float(jensenshannon(a, b, base=2) ** 2)  # scipy returns distance -> square for divergence
    kl = float(kl_entropy(a + EPS, b + EPS))
    emd = float(ot.emd2(a, b, cost))
    s2d = ssim(a.reshape(side, side), b.reshape(side, side), data_range=max(a.max(), b.max()) - min(a.min(), b.min()))
    return cos, pear, spear, js, kl, emd, float(s2d)


def dirauc(v, p, ne):
    ok = ~np.isnan(v)
    y = np.r_[np.ones((p & ok).sum()), np.zeros((ne & ok).sum())]
    s = np.r_[v[p & ok], v[ne & ok]]
    if len(np.unique(y)) < 2 or len(np.unique(s)) < 2:
        return np.nan
    r = roc_auc_score(y, s)
    return r if r >= 0.5 else 1 - r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=f"{REPO}/attn_maps/trajectory_dof_attention.npz")
    ap.add_argument("--steps", type=int, default=10, help="number of leading control steps to use")
    args = ap.parse_args()

    d = np.load(args.npz)
    maps, roles = d["maps"], d["role"]
    n, T, dof, npatch = maps.shape
    assert dof == N_DOF
    side = int(round(np.sqrt(npatch)))
    assert side * side == npatch, f"num_patches={npatch} is not a perfect square"
    steps = min(args.steps, T)
    n_trans = steps - 1
    cost = ot.dist(grid_coords(side), grid_coords(side), metric="euclidean")

    maps = maps[:, :steps]  # (n, steps, 7, npatch)
    valid = ~np.isnan(maps[:, :, 0, 0]).any(axis=1)  # episode has all `steps` populated
    print(f"{valid.sum()}/{n} episodes have >= {steps} valid steps", flush=True)

    metric_names = ["cosine", "pearson", "spearman", "js", "kl", "emd", "ssim"]
    out = {m: np.full((n, N_DOF, n_trans), np.nan) for m in metric_names}

    for i in range(n):
        if not valid[i]:
            continue
        for dd in range(N_DOF):
            for t in range(n_trans):
                vals = compute_pair_metrics(maps[i, t, dd], maps[i, t + 1, dd], cost, side)
                for m, v in zip(metric_names, vals):
                    out[m][i, dd, t] = v

    stem = os.path.splitext(os.path.basename(args.npz))[0]
    out_path = f"{REPO}/attn_maps/dof_timeseries_metrics_{stem}.npz"
    np.savez(out_path, role=roles, steps=steps, **out)
    print(f"Saved -> {out_path}\n", flush=True)

    roles_uniq = [r for r in np.unique(roles) if r != "clean"]
    is_c = roles == "clean"

    print("=" * 80)
    print(f"PER-METRIC OVERALL (episode x DoF x transition pooled, n_valid_ep={valid.sum()}, "
          f"steps={steps}, transitions={n_trans}, dof={N_DOF})")
    print("=" * 80)
    for m in metric_names:
        v = out[m][valid]
        print(f"  {m:<10} mean={np.nanmean(v):.4f}  std={np.nanstd(v):.4f}  "
              f"range=[{np.nanmin(v):.4f},{np.nanmax(v):.4f}]")

    print("\nPER-DoF (mean over episodes & transitions):")
    header = "  DoF  " + "".join(f"{m:>10}" for m in metric_names)
    print(header)
    for dd in range(N_DOF):
        row = "  " + f"{dd+1:<5}" + "".join(f"{np.nanmean(out[m][valid, dd]):>10.4f}" for m in metric_names)
        print(row)

    print(f"\nTIME TREND (mean over episodes & DoF, per transition t->t+1, t=0..{n_trans-1}):")
    for m in metric_names:
        vals = [np.nanmean(out[m][valid, :, t]) for t in range(n_trans)]
        print(f"  {m:<10} " + " ".join(f"{v:.3f}" for v in vals))

    print("\nPER-ROLE (episode-level mean over DoF & transitions) + poison/attack-vs-clean AUROC:")
    ep_mean = {m: np.nanmean(out[m], axis=(1, 2)) for m in metric_names}
    for m in metric_names:
        v = ep_mean[m]
        line = f"  {m:<10}"
        for role in ["clean"] + roles_uniq:
            msk = roles == role
            line += f" {role}={np.nanmean(v[msk & valid]):.4f}"
        for role in roles_uniq:
            auc = dirauc(v, roles == role, is_c)
            line += f"  AUROC[{role}]={auc:.3f}"
        print(line)


if __name__ == "__main__":
    main()
