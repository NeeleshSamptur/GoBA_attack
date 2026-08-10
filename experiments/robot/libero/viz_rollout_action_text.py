"""
Visualize action ↔ text attention rollouts.

Usage:
  python viz_rollout_action_text.py badvla
  python viz_rollout_action_text.py goba
"""
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = "/home/grads/nsamptur/vla_bkd_def/GoBA_attack/attn_maps/single_sample_analysis"


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
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


def lab(t):
    s = str(t).replace("▁", " ").replace("Ġ", " ").replace("<0x0A>", "\\n").strip()
    return s or "∅"


def viz(which):
    ddir = f"{ROOT}/{which}_rollout_action_text_xattn"
    d = np.load(f"{ddir}/rollout_action_text.npz", allow_pickle=True)
    Ac, Ap = d["clean_action2text"], d["poison_action2text"]
    dofs = [str(x) for x in d["dof_names"]]
    toks = [lab(t) for t in d["tokens"]]
    prompt = str(d["prompt"])
    T = min(Ac.shape[0], Ap.shape[0])
    L = min(Ac.shape[2], Ap.shape[2], len(toks))
    toks = toks[:L]
    Ac, Ap = Ac[:T, :, :L], Ap[:T, :, :L]
    n_dof = Ac.shape[1]

    vmin = min(np.percentile(Ac, 1), np.percentile(Ap, 1))
    vmax = max(np.percentile(Ac, 99), np.percentile(Ap, 99))

    # 1) MATRIX GIF: action DoF × text token
    frames = []
    for t in range(T):
        fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.45 * L + 4), 4.2))
        for ax, M, title in [
            (axes[0], Ac[t], "CLEAN × clean scene"),
            (axes[1], Ap[t], "BACKDOORED × TRIGGER"),
        ]:
            im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            ax.set_yticks(range(n_dof))
            ax.set_yticklabels(dofs, fontsize=9)
            ax.set_xticks(range(L))
            ax.set_xticklabels(toks, rotation=75, ha="right", fontsize=7)
            ax.set_title(f"{title}\nmax={M.max():.3f}  H={(-(M*np.log(M+1e-12)).sum(1).mean()):.2f}")
        fig.colorbar(im, ax=axes.tolist(), shrink=0.75, label="action→text attn")
        fig.suptitle(f"{which.upper()} ACTION→TEXT  t={t}/{T-1}\n{prompt.strip()}", fontsize=11)
        frames.append(fig_to_pil(fig)); plt.close(fig)
    frames = pad_frames(frames)
    frames[0].save(f"{ddir}/GIF_action2text_MATRIX.gif", save_all=True,
                   append_images=frames[1:], duration=700, loop=0)
    print("saved GIF_action2text_MATRIX.gif")

    # 2) STATIC: t=0 side-by-side + mean over steps
    fig, axes = plt.subplots(2, 2, figsize=(max(11, 0.45 * L + 5), 8))
    for ax, M, title in [
        (axes[0, 0], Ac[0], "t=0 CLEAN"),
        (axes[0, 1], Ap[0], "t=0 BACKDOOR+TRIGGER"),
        (axes[1, 0], Ac.mean(0), "mean_t CLEAN"),
        (axes[1, 1], Ap.mean(0), "mean_t BACKDOOR+TRIGGER"),
    ]:
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_yticks(range(n_dof)); ax.set_yticklabels(dofs, fontsize=8)
        ax.set_xticks(range(L)); ax.set_xticklabels(toks, rotation=75, ha="right", fontsize=6)
        ax.set_title(title)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="action→text")
    fig.suptitle(f"{which.upper()} action→text summary\n{prompt.strip()}")
    fig.savefig(f"{ddir}/STATIC_action2text.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved STATIC_action2text.png")

    # 3) Per-DoF bar: which text tokens get mass at t=0
    fig, axes = plt.subplots(n_dof, 2, figsize=(max(10, 0.4 * L + 3), 1.6 * n_dof),
                             sharex=True, sharey=True)
    x = np.arange(L)
    for i, dof in enumerate(dofs):
        axes[i, 0].bar(x, Ac[0, i], color="steelblue")
        axes[i, 1].bar(x, Ap[0, i], color="darkorange")
        axes[i, 0].set_ylabel(dof, fontsize=8)
        if i == 0:
            axes[i, 0].set_title("CLEAN t=0")
            axes[i, 1].set_title("BACKDOOR+TRIGGER t=0")
    axes[-1, 0].set_xticks(x); axes[-1, 0].set_xticklabels(toks, rotation=75, ha="right", fontsize=6)
    axes[-1, 1].set_xticks(x); axes[-1, 1].set_xticklabels(toks, rotation=75, ha="right", fontsize=6)
    fig.suptitle(f"{which.upper()} action→text mass per DoF (t=0)")
    fig.tight_layout()
    fig.savefig(f"{ddir}/t0_per_dof_bars.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved t0_per_dof_bars.png")

    # 4) Entropy over time
    Hc = (-(Ac * np.log(Ac + 1e-12)).sum(-1)).mean(-1)
    Hp = (-(Ap * np.log(Ap + 1e-12)).sum(-1)).mean(-1)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(Hc, "o-", label="clean", color="steelblue")
    ax.plot(Hp, "s-", label="backdoor+trigger", color="darkorange")
    ax.set_xlabel("rollout step"); ax.set_ylabel("mean action→text entropy")
    ax.set_title(f"{which.upper()} action→text entropy (higher = flatter over text)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(f"{ddir}/ENTROPY_action2text.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved ENTROPY_action2text.png")

    with open(f"{ddir}/README.txt", "w") as f:
        f.write(f"Mode: {which} action↔text\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"DoFs: {dofs}\n")
        f.write(f"Text tokens: {toks}\n")
        f.write(f"Shapes: clean={Ac.shape} poison={Ap.shape}\n")
        f.write("Primary: action DoF → text token attention (row-normalized).\n")
        f.write("Causal LM: text→action is usually masked/empty.\n")
    print(f"Done -> {ddir}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "badvla"
    assert which in ("badvla", "goba")
    viz(which)
