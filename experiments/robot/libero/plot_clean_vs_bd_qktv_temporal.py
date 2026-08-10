#!/usr/bin/env python3
"""Compare clean vs backdoored QKTV FTT across timesteps + map strips."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

OUT = Path("/home/grads/nsamptur/vla_bkd_def/GoBA_attack/attn_maps/action_imgtext_qktv_temporal")
GRID = 16

PAIRS = [
    ("clean_goba_protocol", "goba", "Clean OpenVLA", "GoBA"),
    ("clean_badvla_protocol", "badvla", "Clean OFT", "BadVLA"),
]


def load(stem):
    p = OUT / f"{stem}_temporal.npz"
    d = np.load(p, allow_pickle=True)
    return {
        "task_id": d["task_id"],
        "cond": np.array([str(c) for c in d["cond"]]),
        "ftt_q": d["ftt_qktv"],
        "maps_q": d["maps_qktv"],
        "rgb": d["rgb"],
    }


def auroc_mean(ftt, cond):
    scores = np.nanmean(ftt, axis=1)
    y = (cond == "poison").astype(int)
    try:
        return float(roc_auc_score(y, -scores))
    except Exception:
        return float("nan")


def overlay_map(rgb, flat, alpha=0.55):
    m = flat.reshape(GRID, GRID)
    m = m - np.nanmin(m)
    m = m / (np.nanmax(m) + 1e-12)
    scale = 224 // GRID
    m_up = np.kron(m, np.ones((scale, scale)))
    cmap = plt.cm.inferno(m_up)[..., :3]
    base = rgb.astype(np.float32) / 255.0
    return (np.clip((1 - alpha) * base + alpha * cmap, 0, 1) * 255).astype(np.uint8)


def plot_ftt_compare(clean, bd, cname, bname, tag):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    T = clean["ftt_q"].shape[1]
    xs = np.arange(T)
    for col, (data, name, color) in enumerate(
        ((clean, cname, "#2a6f97"), (bd, bname, "#c1121f"))
    ):
        for row, cond in enumerate(("clean", "poison")):
            ax = axes[row, col]
            idx = np.where(data["cond"] == cond)[0]
            for i in idx:
                ax.plot(xs, data["ftt_q"][i], color=color, alpha=0.3, lw=1)
            mean = np.nanmean(data["ftt_q"][idx], axis=0)
            ax.plot(xs, mean, color=color, lw=2.5, label="mean")
            ax.set_title(f"{name} · {cond}")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, T - 1)
            if col == 0:
                ax.set_ylabel("FTT (QKTV)")
            if row == 1:
                ax.set_xlabel("timestep")
    fig.suptitle(f"QKTV FTT across timesteps · {cname} vs {bname}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f"FIGURE_{tag}_clean_vs_bd_ftt_qktv_vs_t.png", dpi=160)
    plt.close(fig)


def plot_strips_compare(clean, bd, cname, bname, tag, n_show=4):
    """For each of n_show tasks: 4 rows = clean_model clean/poison, bd clean/poison."""
    tasks = sorted(set(int(t) for t in clean["task_id"]))[:n_show]
    T = clean["maps_q"].shape[1]
    nrows = len(tasks) * 4
    fig, axes = plt.subplots(nrows, T, figsize=(1.35 * T, 1.35 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    row_specs = [
        (clean, "clean", f"{cname}\nclean"),
        (clean, "poison", f"{cname}\npoison"),
        (bd, "clean", f"{bname}\nclean"),
        (bd, "poison", f"{bname}\npoison"),
    ]
    for ti, tid in enumerate(tasks):
        for j, (data, cond, ylab) in enumerate(row_specs):
            r = ti * 4 + j
            idx = np.where((data["task_id"] == tid) & (data["cond"] == cond))[0]
            if len(idx) == 0:
                for t in range(T):
                    axes[r, t].axis("off")
                continue
            i = idx[0]
            for t in range(T):
                ax = axes[r, t]
                if np.isnan(data["maps_q"][i, t]).all():
                    ax.axis("off")
                    continue
                ax.imshow(overlay_map(data["rgb"][i, t], data["maps_q"][i, t]))
                ax.set_xticks([])
                ax.set_yticks([])
                if t == 0:
                    ax.set_ylabel(f"t{tid} {ylab}", fontsize=7)
                if r == 0:
                    ax.set_title(f"t={t}", fontsize=8)
    fig.suptitle(f"mean-DoF QKTV action→image · {cname} vs {bname}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"FIGURE_{tag}_clean_vs_bd_map_strips.png", dpi=130)
    plt.close(fig)


def main():
    lines = [
        "# Clean vs backdoored QKTV across timesteps",
        "",
        "Protocol: LIBERO-Goal 0–9, seed=7, T=10, action→(image+text) QKTV FTT.",
        "",
        "| protocol | clean model AUROC | backdoored AUROC |",
        "|---|---:|---:|",
    ]
    for cstem, bstem, cname, bname in PAIRS:
        cp, bp = OUT / f"{cstem}_temporal.npz", OUT / f"{bstem}_temporal.npz"
        if not cp.exists() or not bp.exists():
            print(f"missing {cp.name if not cp.exists() else bp.name}")
            continue
        clean, bd = load(cstem), load(bstem)
        tag = "goba" if "goba" in bstem else "badvla"
        plot_ftt_compare(clean, bd, cname, bname, tag)
        plot_strips_compare(clean, bd, cname, bname, tag, n_show=4)
        ac = auroc_mean(clean["ftt_q"], clean["cond"])
        ab = auroc_mean(bd["ftt_q"], bd["cond"])
        lines.append(f"| {cname} vs {bname} | {ac:.3f} | {ab:.3f} |")
        print(f"{cname}: AUROC={ac:.3f}  {bname}: AUROC={ab:.3f}")
    lines += [
        "",
        "Clean-model AUROC near 0.5 ⇒ trigger does not collapse QKTV FTT.",
        "Backdoored AUROC high ⇒ poison FTT drops vs clean scenes.",
        "",
        "## Files",
        "- `FIGURE_*_clean_vs_bd_ftt_qktv_vs_t.png`",
        "- `FIGURE_*_clean_vs_bd_map_strips.png`",
        "- `clean_*_protocol_temporal.npz`",
        "",
    ]
    (OUT / "SUMMARY_clean_vs_bd.md").write_text("\n".join(lines))
    print(f"Wrote {OUT}/SUMMARY_clean_vs_bd.md")


if __name__ == "__main__":
    main()
