#!/usr/bin/env python3
"""Temporal similarity of action→image attention maps across rollout timesteps.

Loads existing rollout NPZs (T, n_dof, 256), takes first --num_steps, and for
clean vs backdoored computes:
  cosine, JS divergence, 2D Wasserstein (EMD), SSIM, Pearson, Spearman

Comparisons:
  consecutive: metric(map_t, map_{t+1})
  vs_t0:       metric(map_t, map_0)

Spatial metrics use mean-over-DoF 16×16 maps (normalized to sum=1 where needed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr, wasserstein_distance_nd
from skimage.metrics import structural_similarity as ssim


def _to_prob(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    s = x.sum()
    if s <= eps:
        return np.full_like(x, 1.0 / x.size)
    return x / s


def _mean_spatial(maps_tdp: np.ndarray, gh: int = 16, gw: int = 16) -> np.ndarray:
    """(T, D, P) -> (T, H, W) mean over DoF, then per-timestep L1-normalize."""
    t, d, p = maps_tdp.shape
    assert p == gh * gw
    spat = maps_tdp.reshape(t, d, gh, gw).mean(axis=1)
    out = np.zeros_like(spat, dtype=np.float64)
    for i in range(t):
        out[i] = _to_prob(spat[i])
    return out


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    af, bf = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(af), np.linalg.norm(bf)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(af, bf) / (na * nb))


def js_div(a: np.ndarray, b: np.ndarray) -> float:
    """Jensen–Shannon divergence in nats (scipy returns distance = sqrt(JS);
    we report squared JS so it is a true divergence in [0, ln2])."""
    pa, pb = _to_prob(a).ravel(), _to_prob(b).ravel()
    # jensenshannon returns sqrt(JS) with base=e by default → JS = dist**2
    return float(jensenshannon(pa, pb, base=np.e) ** 2)


def emd_2d(a: np.ndarray, b: np.ndarray) -> float:
    """2D Wasserstein-1 between discrete distributions on the same grid."""
    pa, pb = _to_prob(a), _to_prob(b)
    h, w = pa.shape
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float64)
    return float(
        wasserstein_distance_nd(coords, coords, u_weights=pa.ravel(), v_weights=pb.ravel())
    )


def ssim_map(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = _to_prob(a), _to_prob(b)
    dr = float(max(pa.max(), pb.max()) - min(pa.min(), pb.min()))
    if dr < 1e-12:
        return 1.0
    return float(ssim(pa, pb, data_range=dr))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    r, _ = pearsonr(a.ravel(), b.ravel())
    return float(r) if np.isfinite(r) else 0.0


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    r, _ = spearmanr(a.ravel(), b.ravel())
    return float(r) if np.isfinite(r) else 0.0


METRICS = {
    "cosine": cosine_sim,  # higher = more similar
    "js": js_div,  # lower = more similar
    "emd": emd_2d,  # lower = more similar
    "ssim": ssim_map,  # higher = more similar
    "pearson": pearson_corr,  # higher = more similar
    "spearman": spearman_corr,  # higher = more similar
}

SIMILARITY_HIGHER = {"cosine", "ssim", "pearson", "spearman"}


def pairwise_series(maps: np.ndarray, mode: str) -> dict[str, np.ndarray]:
    """maps: (T, H, W). mode in {consecutive, vs_t0}."""
    t = maps.shape[0]
    out = {k: [] for k in METRICS}
    if mode == "consecutive":
        pairs = [(i, i + 1) for i in range(t - 1)]
    elif mode == "vs_t0":
        pairs = [(0, i) for i in range(1, t)]
    else:
        raise ValueError(mode)
    for i, j in pairs:
        for name, fn in METRICS.items():
            out[name].append(fn(maps[i], maps[j]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def analyze_npz(path: Path, num_steps: int) -> dict:
    d = np.load(path, allow_pickle=True)
    clean = np.asarray(d["clean_text2patch"], dtype=np.float64)[:num_steps]
    poison = np.asarray(d["poison_text2patch"], dtype=np.float64)[:num_steps]
    assert clean.shape[0] >= num_steps and poison.shape[0] >= num_steps
    clean_s = _mean_spatial(clean)
    poison_s = _mean_spatial(poison)
    result = {
        "source": str(path),
        "num_steps": num_steps,
        "map_shape_hw": list(clean_s.shape[1:]),
        "n_dof": int(clean.shape[1]),
        "clean": {},
        "poison": {},
    }
    for mode in ("consecutive", "vs_t0"):
        result["clean"][mode] = {k: v.tolist() for k, v in pairwise_series(clean_s, mode).items()}
        result["poison"][mode] = {k: v.tolist() for k, v in pairwise_series(poison_s, mode).items()}
    # also store mean maps for optional viz
    result["_arrays"] = {"clean": clean_s, "poison": poison_s}
    return result


def plot_suite(result: dict, out_dir: Path, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_order = ["cosine", "js", "emd", "ssim", "pearson", "spearman"]
    ylab = {
        "cosine": "cosine similarity ↑",
        "js": "JS divergence ↓",
        "emd": "EMD (Wasserstein-1) ↓",
        "ssim": "SSIM ↑",
        "pearson": "Pearson r ↑",
        "spearman": "Spearman ρ ↑",
    }

    for mode in ("consecutive", "vs_t0"):
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), sharex=True)
        axes = axes.ravel()
        x = np.arange(1, result["num_steps"])  # pair index / t for vs_t0
        if mode == "consecutive":
            xlabel = "timestep pair (t → t+1)"
            xticklabels = [f"{i}-{i+1}" for i in range(result["num_steps"] - 1)]
        else:
            xlabel = "timestep t (vs t=0)"
            xticklabels = [str(i) for i in range(1, result["num_steps"])]

        for ax, m in zip(axes, metric_order):
            yc = np.asarray(result["clean"][mode][m])
            yp = np.asarray(result["poison"][mode][m])
            ax.plot(x, yc, "o-", color="#2ca02c", label="clean", lw=2, ms=5)
            ax.plot(x, yp, "s--", color="#d62728", label="backdoored", lw=2, ms=5)
            ax.set_title(m)
            ax.set_ylabel(ylab[m])
            ax.grid(True, alpha=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=8)
            # mean annotation
            ax.text(
                0.02,
                0.02,
                f"μ clean={yc.mean():.3g}\nμ bd={yp.mean():.3g}",
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75, ec="0.8"),
            )
        axes[0].legend(loc="best", fontsize=9)
        fig.suptitle(f"{title}\n{mode.replace('_', ' ')} action→image temporal metrics", fontsize=13)
        fig.supxlabel(xlabel)
        fig.tight_layout()
        fig.savefig(out_dir / f"METRICS_{mode}.png", dpi=160)
        plt.close(fig)

    # compact 2-panel: JS + EMD (recommended pair)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, mode in zip(axes, ("consecutive", "vs_t0")):
        for m, color, ls in (("js", "#1f77b4", "-"), ("emd", "#ff7f0e", "--")):
            yc = np.asarray(result["clean"][mode][m])
            yp = np.asarray(result["poison"][mode][m])
            x = np.arange(len(yc))
            ax.plot(x, yc, ls, color=color, marker="o", label=f"clean {m}", lw=2, ms=4)
            ax.plot(x, yp, ls, color=color, marker="s", alpha=0.55, label=f"bd {m}", lw=2, ms=4)
        ax.set_title(mode.replace("_", " "))
        ax.set_xlabel("pair index" if mode == "consecutive" else "t (vs 0)")
        ax.set_ylabel("distance (↓ more stable)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"{title}: JS + EMD (primary drift metrics)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "METRICS_js_emd_primary.png", dpi=160)
    plt.close(fig)

    # spatial strip: maps at t=0, mid, last
    maps_c = result["_arrays"]["clean"]
    maps_p = result["_arrays"]["poison"]
    t = maps_c.shape[0]
    idxs = [0, t // 2, t - 1]
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.5))
    vmax = max(maps_c[idxs].max(), maps_p[idxs].max())
    for col, ti in enumerate(idxs):
        axes[0, col].imshow(maps_c[ti], cmap="magma", vmin=0, vmax=vmax)
        axes[0, col].set_title(f"clean t={ti}")
        axes[0, col].axis("off")
        axes[1, col].imshow(maps_p[ti], cmap="magma", vmin=0, vmax=vmax)
        axes[1, col].set_title(f"backdoored t={ti}")
        axes[1, col].axis("off")
    fig.suptitle(f"{title}: mean-DoF action→image maps (sum-normalized)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "MAPS_timestep_strip.png", dpi=160)
    plt.close(fig)


def summarize(result: dict) -> str:
    lines = [
        f"source: {result['source']}",
        f"steps: {result['num_steps']}, spatial: {result['map_shape_hw']}, n_dof: {result['n_dof']}",
        "",
        "Interpretation: higher cosine/SSIM/corr = more similar; lower JS/EMD = more similar.",
        "If backdoored JS/EMD stay flatter/lower than clean → attention more frozen/stable.",
        "If backdoored JS/EMD rise more → more drift / instability.",
        "",
    ]
    for mode in ("consecutive", "vs_t0"):
        lines.append(f"=== {mode} (mean over pairs) ===")
        lines.append(f"{'metric':12s} {'clean':>10s} {'backdoor':>10s} {'bd-clean':>10s} note")
        for m in ("cosine", "js", "emd", "ssim", "pearson", "spearman"):
            mc = float(np.mean(result["clean"][mode][m]))
            mp = float(np.mean(result["poison"][mode][m]))
            delta = mp - mc
            if m in SIMILARITY_HIGHER:
                note = "bd more similar" if delta > 0 else "bd less similar"
            else:
                note = "bd more similar" if delta < 0 else "bd less similar"
            lines.append(f"{m:12s} {mc:10.4g} {mp:10.4g} {delta:10.4g}  {note}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--num_steps", type=int, default=10)
    ap.add_argument("--title", type=str, default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    result = analyze_npz(Path(args.npz), args.num_steps)
    title = args.title or Path(args.out_dir).name
    plot_suite(result, out_dir, title)
    text = summarize(result)
    (out_dir / "SUMMARY.txt").write_text(text)
    # json without arrays
    payload = {k: v for k, v in result.items() if k != "_arrays"}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    # save arrays
    np.savez_compressed(
        out_dir / "spatial_maps.npz",
        clean=result["_arrays"]["clean"],
        poison=result["_arrays"]["poison"],
    )
    print(text)
    print(f"\nWrote plots + summary to {out_dir}")


if __name__ == "__main__":
    main()
