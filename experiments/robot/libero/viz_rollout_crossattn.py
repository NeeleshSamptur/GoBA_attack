"""
Visualize rollout cross-attention matrices (text OR action <-> image patches).

Usage:
  python viz_rollout_crossattn.py goba
  python viz_rollout_crossattn.py badvla
  python viz_rollout_crossattn.py goba_action
  python viz_rollout_crossattn.py badvla_action
  python viz_rollout_crossattn.py goba_vnorm
  python viz_rollout_crossattn.py badvla_vnorm
  python viz_rollout_crossattn.py goba_action_vnorm
  python viz_rollout_crossattn.py badvla_action_vnorm
"""
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

ROOT = "/home/grads/nsamptur/vla_bkd_def/GoBA_attack/attn_maps/single_sample_analysis"
GRID = 16
VALID = (
    "goba", "badvla", "goba_action", "badvla_action",
    "goba_vnorm", "badvla_vnorm", "goba_action_vnorm", "badvla_action_vnorm",
)


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def pad_frames(frames):
    W = max(f.width for f in frames)
    H = max(f.height for f in frames)
    out = []
    for f in frames:
        c = Image.new("RGB", (W, H), (255, 255, 255))
        c.paste(f, ((W - f.width) // 2, (H - f.height) // 2))
        out.append(c)
    return out


def tok_labels(toks):
    return [str(t).replace("▁", " ").replace("Ġ", " ").replace("<0x0A>", "\\n").strip() or "∅"
            for t in toks]


def viz(attack):
    vnorm = attack.endswith("_vnorm")
    base = attack[:-6] if vnorm else attack
    suffix = "_vnorm" if vnorm else ""
    if base.endswith("_action"):
        family = base.replace("_action", "")
        ddir = f"{ROOT}/{family}_rollout_action_xattn{suffix}"
        kind = "ACTION→image" + ("  (α·‖Wov‖)" if vnorm else "")
        row_name = "action DoF"
    else:
        ddir = f"{ROOT}/{base}_rollout_xattn{suffix}"
        kind = "TEXT→image" + ("  (α·‖Wov‖)" if vnorm else "")
        row_name = "text token"

    d = np.load(f"{ddir}/rollout_xattn.npz", allow_pickle=True)
    Tc = d["clean_text2patch"]
    Tp = d["poison_text2patch"]
    Pc = d["clean_patch2text"]
    Pp = d["poison_patch2text"]
    Rc, Rp = d["clean_rgb"], d["poison_rgb"]
    labs = tok_labels(d["tokens"] if "tokens" in d.files else d["tokens_clean"])
    prompt = str(d["prompt"])
    T = min(Tc.shape[0], Tp.shape[0])
    L = min(Tc.shape[1], Tp.shape[1], len(labs))
    labs = labs[:L]
    Tc, Tp = Tc[:T, :L], Tp[:T, :L]
    Pc, Pp = Pc[:T, :, :L], Pp[:T, :, :L]
    has_p2q = float(np.nanmax(np.abs(Pc))) > 1e-8  # GoBA action leaves placeholder zeros

    vmin_t = min(np.percentile(Tc, 1), np.percentile(Tp, 1))
    vmax_t = max(np.percentile(Tc, 99), np.percentile(Tp, 99))

    # 1) query→patch MATRIX gif
    frames = []
    for t in range(T):
        fig, axes = plt.subplots(1, 2, figsize=(12, max(3.5, 0.35 * L + 2)))
        for ax, M, title in [
            (axes[0], Tc[t], "CLEAN model × clean scene"),
            (axes[1], Tp[t], "BACKDOORED × TRIGGER scene"),
        ]:
            im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=vmin_t, vmax=vmax_t,
                           interpolation="nearest")
            ax.set_yticks(range(L))
            ax.set_yticklabels([f"{i}:{labs[i]}" for i in range(L)], fontsize=8)
            ax.set_xlabel("image patch index (0..255)")
            ax.set_title(f"{title}\nmax={M.max():.3f}  f_norm="
                         f"{np.linalg.norm(M - M.mean(0), axis=1).mean():.4f}")
        fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label=f"{row_name}→patch attn")
        fig.suptitle(f"{attack.upper()}  {kind} MATRIX  |  t={t}/{T-1}\n{prompt.strip()}", fontsize=11)
        frames.append(fig_to_pil(fig)); plt.close(fig)
    frames = pad_frames(frames)
    frames[0].save(f"{ddir}/GIF_text2patch_MATRIX.gif", save_all=True,
                   append_images=frames[1:], duration=700, loop=0)
    print("saved GIF_text2patch_MATRIX.gif")

    # 2) patch→query MATRIX gif (skip if placeholder)
    if has_p2q:
        vmin_p = min(np.percentile(Pc, 1), np.percentile(Pp, 1))
        vmax_p = max(np.percentile(Pc, 99), np.percentile(Pp, 99))
        frames = []
        for t in range(T):
            fig, axes = plt.subplots(1, 2, figsize=(10, max(3.5, 0.35 * L + 2)))
            for ax, M, title in [
                (axes[0], Pc[t].T, "CLEAN model × clean scene"),
                (axes[1], Pp[t].T, "BACKDOORED × TRIGGER scene"),
            ]:
                im = ax.imshow(M, aspect="auto", cmap="magma", vmin=vmin_p, vmax=vmax_p,
                               interpolation="nearest")
                ax.set_yticks(range(L))
                ax.set_yticklabels([f"{i}:{labs[i]}" for i in range(L)], fontsize=8)
                ax.set_xlabel("image patch index (0..255)")
                ax.set_title(f"{title}  (patch→query, transposed)\nmax={M.max():.3f}")
            fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="patch→query attn")
            fig.suptitle(f"{attack.upper()}  image→query  |  t={t}/{T-1}\n{prompt.strip()}", fontsize=11)
            frames.append(fig_to_pil(fig)); plt.close(fig)
        frames = pad_frames(frames)
        frames[0].save(f"{ddir}/GIF_patch2text_MATRIX.gif", save_all=True,
                       append_images=frames[1:], duration=700, loop=0)
        print("saved GIF_patch2text_MATRIX.gif")
    else:
        print("skipped GIF_patch2text_MATRIX.gif (no patch→action weights for this mode)")

    # 3) overlay gif
    frames = []
    for t in range(T):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        for ax, M, rgb, title in [
            (axes[0], Tc[t].mean(0), Rc[t], "CLEAN"),
            (axes[1], Tp[t].mean(0), Rp[t], "TRIGGER / BACKDOORED"),
        ]:
            up = zoom(M.reshape(GRID, GRID), rgb.shape[0] / GRID, order=1)
            ax.imshow(rgb); im = ax.imshow(up, cmap="jet", alpha=0.55)
            ax.set_title(f"{title}  t={t}  max={M.max():.3f}"); ax.axis("off")
        fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
        fig.suptitle(f"{attack.upper()} mean {kind} overlay | t={t}/{T-1}\n{prompt.strip()}")
        frames.append(fig_to_pil(fig)); plt.close(fig)
    frames = pad_frames(frames)
    frames[0].save(f"{ddir}/GIF_text2patch_OVERLAY.gif", save_all=True,
                   append_images=frames[1:], duration=700, loop=0)
    print("saved GIF_text2patch_OVERLAY.gif")

    # 4) static timesteps
    picks = sorted(set([0, min(5, T-1), min(10, T-1), min(15, T-1), T-1]))
    fig, axes = plt.subplots(2, len(picks), figsize=(3.2 * len(picks), max(4, 0.45 * L + 2)))
    for j, t in enumerate(picks):
        for row, M, title in [(0, Tc[t], "CLEAN"), (1, Tp[t], "BACKDOORED+TRIG")]:
            ax = axes[row, j]
            im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=vmin_t, vmax=vmax_t)
            ax.set_title(f"{title} t={t}", fontsize=9)
            if j == 0:
                ax.set_yticks(range(L)); ax.set_yticklabels([labs[i] for i in range(L)], fontsize=7)
            else:
                ax.set_yticks([])
            ax.set_xticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, label="query→patch")
    fig.suptitle(f"{attack.upper()} {kind} matrices across rollout\n{prompt.strip()}")
    fig.savefig(f"{ddir}/STATIC_text2patch_timesteps.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved STATIC_text2patch_timesteps.png")

    # 5) t=0 distance from mean
    Mc0, Mp0 = Tc[0], Tp[0]
    dist_c = np.linalg.norm(Mc0 - Mc0.mean(0), axis=1)
    dist_p = np.linalg.norm(Mp0 - Mp0.mean(0), axis=1)
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * L + 2), 4.0))
    x = np.arange(L); w = 0.38
    ax.bar(x - w/2, dist_c, w, label=f"CLEAN f_norm={dist_c.mean():.4f}", color="steelblue")
    ax.bar(x + w/2, dist_p, w, label=f"BACKDOORED+TRIG f_norm={dist_p.mean():.4f}", color="crimson")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}:{labs[i]}" for i in range(L)], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel(r"$\|M^{(i)}-\bar M\|_2$")
    ax.set_title(f"{attack.upper()} t=0  {kind}  |  distance of each {row_name} from mean")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{ddir}/t0_distance_from_mean.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved t0_distance_from_mean.png")

    # 6) t=0 all queries + mean overlays
    fig, axes = plt.subplots(L + 1, 2, figsize=(7.5, 1.55 * (L + 1)))
    for col, (M, rgb, dists, top) in enumerate([
        (Mc0, Rc[0], dist_c, "CLEAN"),
        (Mp0, Rp[0], dist_p, "BACKDOORED+TRIG"),
    ]):
        mbar = M.mean(0)
        allv = np.concatenate([M.ravel(), mbar.ravel()])
        vmin, vmax = np.percentile(allv, 1), np.percentile(allv, 99)
        for i in range(L):
            ax = axes[i, col]
            up = zoom(M[i].reshape(GRID, GRID), rgb.shape[0] / GRID, order=1)
            ax.imshow(rgb); ax.imshow(up, cmap="jet", alpha=0.55, vmin=vmin, vmax=vmax)
            ax.set_title(f'[{i}] "{labs[i]}"  dist={dists[i]:.3f}', fontsize=8); ax.axis("off")
        ax = axes[L, col]
        up = zoom(mbar.reshape(GRID, GRID), rgb.shape[0] / GRID, order=1)
        ax.imshow(rgb); im = ax.imshow(up, cmap="jet", alpha=0.55, vmin=vmin, vmax=vmax)
        ax.set_title(f"AVERAGE  f_norm={dists.mean():.4f}", fontsize=9, fontweight="bold"); ax.axis("off")
        axes[0, col].annotate(top, xy=(0.5, 1.35), xycoords="axes fraction",
                              ha="center", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.3)
    fig.suptitle(f"{attack.upper()} t=0  each {row_name} + mean\n{prompt.strip()}")
    fig.savefig(f"{ddir}/t0_all_tokens_and_mean.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved t0_all_tokens_and_mean.png")

    with open(f"{ddir}/README.txt", "w") as f:
        f.write(f"Mode: {attack}\nKind: {kind}\nPrompt: {prompt}\n")
        f.write(f"Query rows ({row_name}): {labs}\n")
        f.write(f"Shapes: clean={Tc.shape} poison={Tp.shape}\n")
        f.write("npz key clean_text2patch holds query→patch (action DoFs when *_action).\n")
        f.write("Row-normalized.\n")
        if "note" in d.files:
            f.write(f"Note: {d['note']}\n")
    print(f"Done -> {ddir}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "goba"
    assert which in VALID, f"got {which}, expected one of {VALID}"
    viz(which)
