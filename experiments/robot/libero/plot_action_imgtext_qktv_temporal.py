#!/usr/bin/env python3
"""Plot temporal QKTV FTT curves + attention-map strips for GoBA / BadVLA."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

OUT = Path("/home/grads/nsamptur/vla_bkd_def/GoBA_attack/attn_maps/action_imgtext_qktv_temporal")
GRID = 16


def load(name):
    p = OUT / f"{name}_temporal.npz"
    d = np.load(p, allow_pickle=True)
    return {
        "task_id": d["task_id"],
        "cond": np.array([str(c) for c in d["cond"]]),
        "ftt_w": d["ftt_weight"],
        "ftt_q": d["ftt_qktv"],
        "maps_w": d["maps_weight"],
        "maps_q": d["maps_qktv"],
        "rgb": d["rgb"],
    }


def auroc_mean(ftt, cond):
    """Episode score = nanmean over T; higher score = more clean-like if poison drops."""
    scores = np.nanmean(ftt, axis=1)
    y = (cond == "poison").astype(int)
    # poison should have LOWER FTT → score for detection = -FTT
    try:
        return float(roc_auc_score(y, -scores))
    except Exception:
        return float("nan")


def plot_ftt_curves(data, name, key, title, fname):
    ftt = data[key]
    cond = data["cond"]
    tasks = data["task_id"]
    T = ftt.shape[1]
    xs = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, c, color in zip(axes, ("clean", "poison"), ("#2a6f97", "#c1121f")):
        idx = np.where(cond == c)[0]
        for i in idx:
            ax.plot(xs, ftt[i], color=color, alpha=0.35, lw=1.2)
            ax.annotate(str(int(tasks[i])), (0, ftt[i, 0]), fontsize=7, color=color, alpha=0.7)
        mean = np.nanmean(ftt[idx], axis=0)
        ax.plot(xs, mean, color=color, lw=2.5, label=f"mean {c}")
        ax.set_title(f"{name} · {c}")
        ax.set_xlabel("timestep")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, T - 1)
    axes[0].set_ylabel(title)
    fig.suptitle(f"{name}: {title} across timesteps (1 demo/task, seed=7)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160)
    plt.close(fig)


def overlay_map(rgb, flat, alpha=0.55):
    """rgb (224,224,3), flat (256,) -> overlay uint8."""
    m = flat.reshape(GRID, GRID)
    m = m - np.nanmin(m)
    m = m / (np.nanmax(m) + 1e-12)
    # upsample nearest to 224
    scale = 224 // GRID
    m_up = np.kron(m, np.ones((scale, scale)))
    cmap = plt.cm.inferno(m_up)[..., :3]
    base = rgb.astype(np.float32) / 255.0
    out = (1 - alpha) * base + alpha * cmap
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def plot_strips(data, name, map_key, fname, n_show=5):
    """For each of first n_show tasks: clean|poison rows × T columns of overlays."""
    cond = data["cond"]
    tasks = data["task_id"]
    maps = data[map_key]
    rgb = data["rgb"]
    T = maps.shape[1]
    uniq = sorted(set(int(t) for t in tasks))[:n_show]

    fig, axes = plt.subplots(len(uniq) * 2, T, figsize=(1.4 * T, 1.5 * len(uniq) * 2))
    if len(uniq) == 1:
        axes = np.array([axes])
    for r, tid in enumerate(uniq):
        for j, c in enumerate(("clean", "poison")):
            row = r * 2 + j
            idx = np.where((tasks == tid) & (cond == c))[0]
            if len(idx) == 0:
                continue
            i = idx[0]
            for t in range(T):
                ax = axes[row, t]
                if np.isnan(maps[i, t]).all():
                    ax.axis("off")
                    continue
                img = overlay_map(rgb[i, t], maps[i, t])
                ax.imshow(img)
                ax.set_xticks([]); ax.set_yticks([])
                if t == 0:
                    ax.set_ylabel(f"t{tid}\n{c}", fontsize=8)
                if row == 0:
                    ax.set_title(f"t={t}", fontsize=8)
    fig.suptitle(f"{name}: mean-DoF QKTV action→image over timesteps", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=140)
    plt.close(fig)


def plot_delta_heatmap(data, name, key, fname):
    """Per-task ΔFTT = clean − poison over time."""
    ftt = data[key]
    cond = data["cond"]
    tasks = data["task_id"]
    uniq = sorted(set(int(t) for t in tasks))
    T = ftt.shape[1]
    mat = np.full((len(uniq), T), np.nan)
    for r, tid in enumerate(uniq):
        ic = np.where((tasks == tid) & (cond == "clean"))[0]
        ip = np.where((tasks == tid) & (cond == "poison"))[0]
        if len(ic) and len(ip):
            mat[r] = ftt[ic[0]] - ftt[ip[0]]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-np.nanmax(np.abs(mat)),
                   vmax=np.nanmax(np.abs(mat)))
    ax.set_yticks(range(len(uniq))); ax.set_yticklabels([f"t{t}" for t in uniq])
    ax.set_xlabel("timestep"); ax.set_title(f"{name}: clean−poison FTT (red=clean higher)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160)
    plt.close(fig)


def summary_md(results):
    lines = [
        "# Action→(image+text) QKTV across timesteps",
        "",
        "Protocol: LIBERO-Goal tasks 0–9, seed=7, T=10 closed-loop steps after settle.",
        "Maps = mean over 7 DoF of QKTV action→image (16×16), overlaid on RGB.",
        "",
        "## AUROC (episode score = −mean_t FTT_QKTV; poison expected lower)",
        "",
        "| model | FTT weight AUROC | FTT QKTV AUROC |",
        "|---|---:|---:|",
    ]
    for name, d in results.items():
        aw = auroc_mean(d["ftt_w"], d["cond"])
        aq = auroc_mean(d["ftt_q"], d["cond"])
        lines.append(f"| {name} | {aw:.3f} | {aq:.3f} |")
    lines += [
        "",
        "## What to look for in the maps",
        "",
        "- **Persistence**: does the poison hotspot stay fixed while clean wanders?",
        "- **Collapse**: does poison concentrate on fewer patches (lower FTT)?",
        "- **Timing**: does separation appear at t=0 or grow over the rollout?",
        "",
        "## Files",
        "",
        "- `FIGURE_*_ftt_qktv_vs_t.png` — per-episode FTT curves",
        "- `FIGURE_*_map_strips_qktv.png` — attention overlays across t",
        "- `FIGURE_*_delta_ftt_heatmap.png` — clean−poison FTT per task×t",
        "- `*_temporal.npz` — raw arrays",
        "",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(lines))


def main():
    results = {}
    for name in ("goba", "badvla"):
        p = OUT / f"{name}_temporal.npz"
        if not p.exists():
            print(f"missing {p}")
            continue
        d = load(name)
        results[name] = d
        label = "GoBA" if name == "goba" else "BadVLA"
        plot_ftt_curves(d, label, "ftt_q", "FTT (QKTV)", f"FIGURE_{name}_ftt_qktv_vs_t.png")
        plot_ftt_curves(d, label, "ftt_w", "FTT (weight)", f"FIGURE_{name}_ftt_weight_vs_t.png")
        plot_delta_heatmap(d, label, "ftt_q", f"FIGURE_{name}_delta_ftt_heatmap.png")
        plot_strips(d, label, "maps_q", f"FIGURE_{name}_map_strips_qktv.png", n_show=5)
        # also full 10-task strip for browsing
        plot_strips(d, label, "maps_q", f"FIGURE_{name}_map_strips_qktv_all.png", n_show=10)
        print(f"{label}: AUROC_qktv={auroc_mean(d['ftt_q'], d['cond']):.3f} "
              f"AUROC_w={auroc_mean(d['ftt_w'], d['cond']):.3f}")
    if results:
        summary_md(results)
        print(f"Wrote figures + SUMMARY.md -> {OUT}")


if __name__ == "__main__":
    main()
