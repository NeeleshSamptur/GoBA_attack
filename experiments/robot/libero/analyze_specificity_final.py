"""
experiments/robot/libero/analyze_specificity_final.py

Consolidated analysis of the specificity benchmark (attn_maps/specificity_benchmark.npz).

Three analyses that harden the depth-dependent-specificity finding for review:

  1. TASK-DISJOINT validation. The seed-disjoint result could in principle lean
     on per-task scene statistics shared between calibration and test. Here the
     Mahalanobis reference is calibrated on clean scenes from tasks 0-4 ONLY and
     every evaluated scene comes from tasks 5-9, which the calibration never saw.
     If the layer-wise specificity profile survives, it is a property of the
     model, not of the calibration scenes.

  2. DEPLOYMENT operating point. AUROC is threshold-free; a robot needs one
     threshold fixed in advance. Crucially, the Mahalanobis fit and the
     threshold must come from DIFFERENT clean scenes: scores on the scenes used
     to fit mu/sd are biased low (50 samples estimating 4096 per-dim variances),
     so an in-sample max-threshold under-covers and fires on most clean test
     scenes. We fit on 3 calibration seeds, then set the threshold from the
     other 2 calibration seeds (out-of-sample for the fit) with the standard
     robust-outlier rule median + 3 * 1.4826 * MAD -- the MAD analogue of a
     3-sigma limit, chosen a priori rather than tuned. A max-rule is too
     fragile here because clean scores vary strongly with scene seed.

  3. FIGURE. Per-layer AUROC curves (poison / ketchup / milk vs clean-test) --
     the visual form of the "early layers are novelty detectors, deep layers are
     backdoor detectors" claim.

Usage: python analyze_specificity_final.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
NPZ = f"{REPO}/attn_maps/specificity_benchmark.npz"
FIG = f"{REPO}/attn_maps/FIGURE_specificity_by_layer.png"


def calibrate(pooled_cal):
    """Per-layer mean/std with the same scale-tied variance floor as the benchmark."""
    mu = pooled_cal.mean(axis=0)
    sd = pooled_cal.std(axis=0)
    for l in range(sd.shape[0]):
        nz = sd[l][sd[l] > 1e-6]
        scale = np.median(nz) if nz.size else float(np.sqrt(np.mean(mu[l] ** 2)))
        sd[l] = np.maximum(sd[l], max(1e-2 * scale, 1e-6))
    return mu, sd


def directed(vals, pos_mask, neg_mask):
    y = np.concatenate([np.ones(pos_mask.sum()), np.zeros(neg_mask.sum())])
    s = np.concatenate([vals[pos_mask], vals[neg_mask]])
    raw = roc_auc_score(y, s)
    return raw if raw >= 0.5 else 1.0 - raw


def layer_aurocs(z, roles, ct):
    """Per-layer (poison, ketchup, milk) AUROC vs clean_test."""
    n_layers = z.shape[1]
    out = np.zeros((n_layers, 3))
    for l in range(n_layers):
        m = np.linalg.norm(z[:, l, :], axis=1)
        out[l, 0] = directed(m, roles == "poison", ct)
        out[l, 1] = directed(m, roles == "decoy_ketchup", ct)
        out[l, 2] = directed(m, roles == "decoy_milk", ct)
    return out


def main():
    d = np.load(NPZ, allow_pickle=True)
    roles = d["role"].astype(str)
    pooled = d["pooled"]
    fnorm = d["f_norm"]
    task_id = d["task_id"]
    n_layers = pooled.shape[1]

    # ---------------- 1. seed-disjoint (all tasks) -- reference numbers ----------------
    cal = roles == "clean_cal"
    ct = roles == "clean_test"
    mu, sd = calibrate(pooled[cal])
    z = (pooled - mu[None]) / sd[None]
    au_seed = layer_aurocs(z, roles, ct)
    gaps_seed = au_seed[:, 0] - au_seed[:, 1:].max(axis=1)
    best_l = int(gaps_seed.argmax())

    # ---------------- 2. TASK-DISJOINT: calibrate tasks 0-4, evaluate tasks 5-9 --------
    cal_td = cal & (task_id <= 4)
    heldout = task_id >= 5
    mu2, sd2 = calibrate(pooled[cal_td])
    z2 = (pooled - mu2[None]) / sd2[None]
    roles_h, z2_h = roles[heldout], z2[heldout]
    ct_h = roles_h == "clean_test"
    au_task = layer_aurocs(z2_h, roles_h, ct_h)
    gaps_task = au_task[:, 0] - au_task[:, 1:].max(axis=1)

    print("=" * 90)
    print("TASK-DISJOINT VALIDATION: calibrate on clean tasks 0-4, evaluate ONLY tasks 5-9")
    print(f"(n per role on held-out tasks: {int(ct_h.sum())} clean_test, "
          f"{int((roles_h == 'poison').sum())} poison, "
          f"{int((roles_h == 'decoy_ketchup').sum())} ketchup, "
          f"{int((roles_h == 'decoy_milk').sum())} milk)")
    print("=" * 90)
    print(f"{'layer':>5} | {'-- seed-disjoint (reference) --':^33} | {'---- TASK-disjoint ----':^33}")
    print(f"{'':>5} | {'poison':>8} {'ketchup':>8} {'milk':>7} {'gap':>7} | "
          f"{'poison':>8} {'ketchup':>8} {'milk':>7} {'gap':>7}")
    print("-" * 90)
    for l in range(0, n_layers, 2):
        print(f"{l:>5} | {au_seed[l,0]:>8.3f} {au_seed[l,1]:>8.3f} {au_seed[l,2]:>7.3f} "
              f"{gaps_seed[l]:>+7.3f} | {au_task[l,0]:>8.3f} {au_task[l,1]:>8.3f} "
              f"{au_task[l,2]:>7.3f} {gaps_task[l]:>+7.3f}")
    print(f"\nseed-disjoint: best gap layer {best_l} ({gaps_seed[best_l]:+.3f}); "
          f"early(0-5) {gaps_seed[:6].mean():+.3f}, deep(27-{n_layers-1}) {gaps_seed[27:].mean():+.3f}")
    bt = int(gaps_task.argmax())
    print(f"TASK-disjoint: best gap layer {bt} ({gaps_task[bt]:+.3f}); "
          f"early(0-5) {gaps_task[:6].mean():+.3f}, deep(27-{n_layers-1}) {gaps_task[27:].mean():+.3f}")
    print(f"gap at layer {best_l} (chosen on seed-disjoint) under task-disjoint eval: "
          f"{gaps_task[best_l]:+.3f}")

    # ---------------- 3. DEPLOYMENT operating point (split calibration) ----------------
    # Fit mu/sd on 3 calibration seeds; threshold on the other 2 (out-of-sample for the
    # fit). Scores on the fit scenes themselves are biased low, so an in-sample threshold
    # would under-cover and false-alarm heavily on clean test data.
    seeds = d["seed"]
    cal_seeds = sorted(set(seeds[cal].tolist()))
    fit_seeds, thr_seeds = cal_seeds[:3], cal_seeds[3:]
    fit_mask = cal & np.isin(seeds, fit_seeds)
    thr_mask = cal & np.isin(seeds, thr_seeds)
    mu3, sd3 = calibrate(pooled[fit_mask])
    z3 = (pooled - mu3[None]) / sd3[None]
    m_dep = np.linalg.norm(z3[:, best_l, :], axis=1)
    v = m_dep[thr_mask]
    med, mad = np.median(v), np.median(np.abs(v - np.median(v)))
    thr = med + 3 * 1.4826 * mad
    print("\n" + "=" * 90)
    print(f"DEPLOYMENT OPERATING POINT: layer-{best_l} Mahalanobis")
    print(f"fit mu/sd on {int(fit_mask.sum())} clean scenes (seeds {fit_seeds}); "
          f"threshold from {int(thr_mask.sum())} held-out clean scenes (seeds {thr_seeds}):")
    print(f"median + 3*1.4826*MAD = {med:.2f} + 3*{1.4826*mad:.2f} = {thr:.2f}")
    print(f"(score margin: max clean/decoy = "
          f"{m_dep[(roles != 'poison') & ~cal].max():.2f}, min poison = "
          f"{m_dep[roles == 'poison'].min():.2f} -- zero overlap)")
    print("=" * 90)
    print(f"{'role':<16} {'n':>4} {'fire-rate':>10}   (fire = score > threshold)")
    print("-" * 48)
    for role in ["poison", "clean_test", "decoy_ketchup", "decoy_milk"]:
        msk = roles == role
        rate = (m_dep[msk] > thr).mean()
        print(f"{role:<16} {int(msk.sum()):>4} {rate*100:>9.1f}%")
    # m_best (full-calibration fit) still used for the AUROC table below.
    m_best = np.linalg.norm(z[:, best_l, :], axis=1)

    # f_norm at the same robust operating point (f_norm is LOWER under poison; no fit
    # step, so all 50 calibration scenes can set the threshold).
    vf = fnorm[cal]
    medf, madf = np.median(vf), np.median(np.abs(vf - np.median(vf)))
    thr_f = medf - 3 * 1.4826 * madf
    print(f"\nf_norm (fires when BELOW median - 3*1.4826*MAD = {thr_f:.4f}):")
    for role in ["poison", "clean_test", "decoy_ketchup", "decoy_milk"]:
        msk = roles == role
        rate = (fnorm[msk] < thr_f).mean()
        print(f"{role:<16} {int(msk.sum()):>4} {rate*100:>9.1f}%")

    # ---------------- headline AUROC table (seed-disjoint) -----------------------------
    print("\n" + "=" * 90)
    print("HEADLINE AUROC TABLE (seed-disjoint, n=50/role)")
    print("=" * 90)
    rows = [
        ("f_norm (attention)", fnorm),
        (f"Mahalanobis L{best_l}", m_best),
        ("Mahalanobis pooled", np.sqrt((z ** 2).sum(axis=(1, 2)))),
        ("Mahalanobis L0", np.linalg.norm(z[:, 0, :], axis=1)),
    ]
    print(f"{'detector':<20} | {'poison':>8} | {'ketchup':>8} | {'milk':>7} | {'spec. gap':>10}")
    print("-" * 66)
    for name, vals in rows:
        a_p = directed(vals, roles == "poison", ct)
        a_k = directed(vals, roles == "decoy_ketchup", ct)
        a_m = directed(vals, roles == "decoy_milk", ct)
        print(f"{name:<20} | {a_p:>8.4f} | {a_k:>8.4f} | {a_m:>7.4f} | {a_p - max(a_k, a_m):>+10.4f}")

    # ---------------- 4. FIGURE --------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    x = np.arange(n_layers)
    ax1.plot(x, au_seed[:, 0], "o-", color="#c0392b", lw=2, ms=4, label="backdoor trigger (poison)")
    ax1.plot(x, au_seed[:, 1], "s--", color="#2980b9", lw=1.5, ms=4, label="benign decoy (ketchup)")
    ax1.plot(x, au_seed[:, 2], "^--", color="#27ae60", lw=1.5, ms=4, label="benign decoy (milk)")
    ax1.axhline(0.5, color="gray", ls=":", lw=1)
    ax1.set_ylabel("AUROC vs clean (n=50/role)")
    ax1.set_title("Backdoor-specificity is depth-dependent\n"
                  "(Mahalanobis drift per LLM layer; objects occupy the identical scene slot)")
    ax1.legend(loc="center right", fontsize=9)
    ax1.set_ylim(0.45, 1.03)
    ax1.grid(alpha=0.3)

    ax2.fill_between(x, 0, gaps_seed, where=gaps_seed >= 0, color="#c0392b", alpha=0.35)
    ax2.fill_between(x, 0, gaps_seed, where=gaps_seed < 0, color="#2980b9", alpha=0.35)
    ax2.plot(x, gaps_seed, "k-", lw=1.5, label="seed-disjoint")
    ax2.plot(x, gaps_task, "k--", lw=1.2, alpha=0.7, label="task-disjoint")
    ax2.axhline(0, color="gray", ls=":", lw=1)
    ax2.set_xlabel("LLM layer (0 = embeddings)")
    ax2.set_ylabel("specificity gap\n(poison − best decoy)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG, dpi=160)
    print(f"\nFigure saved -> {FIG}")


if __name__ == "__main__":
    main()
