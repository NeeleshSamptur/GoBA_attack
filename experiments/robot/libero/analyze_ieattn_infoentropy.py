#!/usr/bin/env python3
"""Analyze InfoEntropy Ieattn scores for GoBA / BadVLA detection."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

OUT = Path("/home/grads/nsamptur/vla_bkd_def/GoBA_attack/attn_maps/infoentropy_ieattn")
GRID = 16
SCORES = [
    ("score_topk_jac_consec", "Top-k Jaccard consec"),
    ("score_topk_jac_t0", "Top-k Jaccard vs t0"),
    ("score_mean_Ie", "mean Ieattn"),
    ("score_max_Ie", "max Ieattn (mean over t)"),
    ("score_spatial_Ie_entropy", "spatial entropy of Ie"),
]


def auroc_both(y, s):
    """Return (best_auroc, poison_higher_bool, auroc_if_higher, auroc_if_lower)."""
    try:
        a_hi = float(roc_auc_score(y, s))
        a_lo = float(roc_auc_score(y, -s))
    except Exception:
        return float("nan"), True, float("nan"), float("nan")
    if a_hi >= a_lo:
        return a_hi, True, a_hi, a_lo
    return a_lo, False, a_hi, a_lo


def overlay(rgb, flat):
    m = flat.reshape(GRID, GRID)
    m = m - np.nanmin(m)
    m = m / (np.nanmax(m) + 1e-12)
    up = np.kron(m, np.ones((224 // GRID, 224 // GRID)))
    cmap = plt.cm.inferno(up)[..., :3]
    base = rgb.astype(np.float32) / 255.0
    return (np.clip((1 - 0.55) * base + 0.55 * cmap, 0, 1) * 255).astype(np.uint8)


def plot_strips(d, name, n_show=4):
    cond = np.array([str(c) for c in d["cond"]])
    tasks = d["task_id"]
    Ie = d["Ie"]
    rgb = d["rgb"]
    uniq = sorted(set(int(t) for t in tasks))[:n_show]
    T = Ie.shape[1]
    fig, axes = plt.subplots(len(uniq) * 2, T, figsize=(1.35 * T, 1.35 * len(uniq) * 2))
    for r0, tid in enumerate(uniq):
        for j, c in enumerate(("clean", "poison")):
            row = r0 * 2 + j
            idx = np.where((tasks == tid) & (cond == c))[0]
            if not len(idx):
                continue
            i = idx[0]
            for t in range(T):
                ax = axes[row, t]
                if np.isnan(Ie[i, t]).all():
                    ax.axis("off")
                    continue
                ax.imshow(overlay(rgb[i, t], Ie[i, t]))
                ax.set_xticks([]); ax.set_yticks([])
                if t == 0:
                    ax.set_ylabel(f"t{tid}\n{c}", fontsize=7)
                if row == 0:
                    ax.set_title(f"t={t}", fontsize=8)
    fig.suptitle(f"{name}: Ieattn (text→image InfoEntropy) over timesteps", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"FIGURE_{name.lower()}_ieattn_strips.png", dpi=130)
    plt.close(fig)


def plot_score_bars(results):
    # results: list of (name, dict score->(auroc, clean_mean, poison_mean, higher))
    metrics = [s[0] for s in SCORES]
    labels = [s[1] for s in SCORES]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (name, tab) in zip(axes, results):
        xs = np.arange(len(metrics))
        vals = [tab[m]["auroc"] for m in metrics]
        colors = ["#2a9d8f" if v >= 0.7 else ("#e9c46a" if v >= 0.55 else "#e76f51") for v in vals]
        ax.bar(xs, vals, color=colors)
        ax.axhline(0.5, color="k", ls="--", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(name)
        ax.set_ylabel("AUROC (best orientation)")
        for x, v, m in zip(xs, vals, metrics):
            arrow = "↑" if tab[m]["higher"] else "↓"
            ax.text(x, v + 0.02, f"{v:.2f}{arrow}", ha="center", fontsize=8)
    fig.suptitle("InfoEntropy-style detectors · shared protocol T=10", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "FIGURE_auroc_ieattn_scores.png", dpi=160)
    plt.close(fig)


def main():
    lines = [
        "# VLA-InfoEntropy-style Ieattn for GoBA / BadVLA detection",
        "",
        "Adapted InfoEntropy: score(w,i)=mean_{l,h} A_{w→i} (last 8 layers); "
        "q(w,i)=A/∑_w A (renorm, not 2nd softmax); Ieattn=1−H/log2|W|.",
        "Top-k=16. T=10, seed=7, tasks 0–9.",
        "",
        "AUROC uses best orientation; ↑/↓ marks whether poison scores higher or lower.",
        "",
    ]
    results_plot = []
    for stem, name in (("goba", "GoBA"), ("badvla", "BadVLA")):
        p = OUT / f"{stem}_ieattn.npz"
        if not p.exists():
            print(f"missing {p}")
            continue
        d = np.load(p, allow_pickle=True)
        cond = np.array([str(c) for c in d["cond"]])
        y = (cond == "poison").astype(int)
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| metric | AUROC | clean | poison | poison has higher score? |")
        lines.append("|---|---:|---:|---:|---|")
        tab = {}
        for key, label in SCORES:
            s = d[key]
            auc, higher, a_hi, a_lo = auroc_both(y, s)
            cm = float(s[cond == "clean"].mean())
            pm = float(s[cond == "poison"].mean())
            tab[key] = dict(auroc=auc, clean=cm, poison=pm, higher=higher, label=label,
                            a_hi=a_hi, a_lo=a_lo)
            lines.append(
                f"| {label} | {auc:.3f} | {cm:.4f} | {pm:.4f} | "
                f"{'yes' if higher else 'no (poison lower)'} |"
            )
            print(f"{name:7s} {label:32s} AUROC={auc:.3f}  "
                  f"clean={cm:.4f} poison={pm:.4f}  poison↑={higher}")
        lines.append("")
        plot_strips(d, name)
        results_plot.append((name, tab))

    if results_plot:
        plot_score_bars(results_plot)
        lines.append("## Shared (both AUROC≥0.70, **same** poison↑/↓ sign)")
        lines.append("")
        shared = []
        for key, label in SCORES:
            signs = [tab[key]["higher"] for _, tab in results_plot]
            aurocs = [tab[key]["auroc"] for _, tab in results_plot]
            if len(set(signs)) == 1 and all(a >= 0.70 for a in aurocs):
                direction = "poison↑" if signs[0] else "poison↓"
                shared.append(
                    f"- **{label}** ({direction}): " + ", ".join(
                        f"{name}={tab[key]['auroc']:.3f}" for name, tab in results_plot)
                )
        if shared:
            lines.extend(shared)
        else:
            lines.append("- No scalar clears 0.70 on **both** with the **same** sign.")
            # still list near-shared
            lines.append("")
            lines.append("Per-attack best (may disagree in sign):")
            for name, tab in results_plot:
                best = max(tab.values(), key=lambda x: x["auroc"])
                lines.append(
                    f"- {name}: {best['label']} AUROC={best['auroc']:.3f} "
                    f"({'poison↑' if best['higher'] else 'poison↓'})"
                )
        lines.append("")
        lines.append(
            "Note: paper’s 2nd softmax over text on causal VLA self-attn gave Ie≈0; "
            "we use q(w,i)=A_{w→i}/∑_{w'}A_{w'→i} on the last 8 layers."
        )
        lines.append("")

    (OUT / "SUMMARY.md").write_text("\n".join(lines))
    print(f"Wrote {OUT}/SUMMARY.md")


if __name__ == "__main__":
    main()
