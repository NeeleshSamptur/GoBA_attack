"""
Per-head VALUE-WEIGHTED (α_h · ||W_O^h v_h||) analysis for GoBA.

Same task/seed as single_sample rollouts (task 7, seed 7). At the first
control frame after wait, for CLEAN model×clean scene and GOBA×trigger:
  - per-head text→image maps  (H, n_text, 256)
  - per-head action→image maps (H, 7, 256)  [teacher-forced after generate]
  - raw-weight per-head maps for comparison

Plots under:
  attn_maps/single_sample_analysis/goba_perhead_vnorm/
"""
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import json
import numpy as np
import tensorflow as tf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from libero.libero import benchmark
from PIL import Image
from scipy.ndimage import zoom
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.libero.run_libero_eval_attentionmap import crop_and_resize
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer

GOBA_CKPT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
CLEAN_CKPT = "openvla/openvla-7b-finetuned-libero-goal"
OUT = f"{REPO}/attn_maps/single_sample_analysis/goba_perhead_vnorm"
BDDL = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
SUITE, TASK_ID, SEED = "libero_goal", 7, 7
NUM_STEPS_WAIT, DEVICE, N_DOF, LAYER, GRID = 10, 0, 7, -1, 16
DOF_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def preprocess(img):
    im = tf.convert_to_tensor(np.array(Image.fromarray(img).convert("RGB")))
    dt = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, dt, saturate=True)
    return np.array(Image.fromarray(im.numpy()).convert("RGB"))


def lab(t):
    s = str(t).replace("▁", " ").replace("Ġ", " ").replace("<0x0A>", "\\n").strip()
    return s or "∅"


def entropy_rows(M):
    p = M / np.clip(M.sum(-1, keepdims=True), 1e-12, None)
    return float((-(p * np.log(p + 1e-12)).sum(-1)).mean())


def fnorm_rows(M):
    p = M / np.clip(M.sum(-1, keepdims=True), 1e-12, None)
    return float(np.linalg.norm(p - p.mean(0), axis=1).mean())


def per_head_maps(A, vh, Wo_l, q_rows, k_cols):
    """Returns:
      vw:  (H, Q, K)  α_h * ||Wo^h v_h||  (raw, not row-norm)
      w:   (H, Q, K)  α_h only
      mass:(H,)       total vw mass over Q×K (head importance)
    """
    f = torch.einsum("nhd,ohd->nho", vh, Wo_l.float())
    f_norms = f.norm(dim=-1)  # (N, H)
    Aq = A[:, q_rows, :][:, :, k_cols]  # (H, Q, K)
    vn = f_norms[k_cols, :].T  # (H, K)
    vw = Aq * vn.unsqueeze(1)  # (H, Q, K)
    w = Aq
    mass = vw.sum(dim=(1, 2))
    return vw.float().cpu().numpy(), w.float().cpu().numpy(), mass.float().cpu().numpy()


def row_norm_h(M):
    """(H,Q,K) -> row-normalize over K per (h,q)."""
    return M / np.clip(M.sum(-1, keepdims=True), 1e-12, None)


def get_scene(bddl_key):
    task = benchmark.get_benchmark_dict()[SUITE]().get_task(TASK_ID)
    env, desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False,
                               bddl_path=BDDL[bddl_key], seed=SEED)
    env.reset(); obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
    img = preprocess(get_libero_image(obs, 224))
    env.close()
    return img, desc


def load_and_probe(ckpt, img, desc):
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE)
    if os.path.isdir(ckpt) and os.path.exists(os.path.join(ckpt, "dataset_statistics.json")):
        vla.norm_stats = json.load(open(os.path.join(ckpt, "dataset_statistics.json")))
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    n_patch = vla.vision_backbone.featurizer.patch_embed.num_patches
    layers = vla.language_model.model.layers
    attn0 = layers[0].self_attn
    n_heads, head_dim = attn0.num_heads, attn0.head_dim
    hidden = n_heads * head_dim
    assert getattr(attn0, "num_key_value_heads", n_heads) == n_heads
    v_cache = {}
    handles = [lyr.self_attn.v_proj.register_forward_hook(
        lambda _m, _i, out, l=l: v_cache.__setitem__(l, out))
        for l, lyr in enumerate(layers)]
    Wo = {l: layers[l].self_attn.o_proj.weight.detach().view(hidden, n_heads, head_dim)
          for l in range(len(layers))}
    li = LAYER if LAYER >= 0 else len(layers) + LAYER

    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, Image.fromarray(img)).to(DEVICE, dtype=torch.bfloat16)

    # TEXT
    v_cache.clear()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(**inputs, output_attentions=True)
    A = out.attentions[li][0].float()
    N = A.shape[-1]
    vh = v_cache[li][0].float().view(N, n_heads, head_dim)
    mask = (inputs.input_ids < action_tokenizer.action_token_begin_idx)[0]
    text_rows = (1 + n_patch + np.nonzero(mask.cpu().numpy()[1:])[0]).tolist()
    patch_cols = list(range(1, 1 + n_patch))
    tw_vw, tw_w, tw_mass = per_head_maps(A, vh, Wo[li], text_rows, patch_cols)
    ids = inputs.input_ids[0].tolist()
    toks = [lab(t) for i, t in enumerate(
        processor.tokenizer.convert_ids_to_tokens(ids)) if i > 0 and bool(mask[i])]
    n = min(len(toks), tw_vw.shape[1])
    toks, tw_vw, tw_w = toks[:n], tw_vw[:, :n], tw_w[:, :n]
    del out

    # ACTION (generate + teacher force)
    input_ids = inputs.input_ids
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        gen = vla.generate(input_ids, pixel_values=inputs.pixel_values,
                           max_new_tokens=N_DOF, do_sample=False)
    action_ids = gen[0, -N_DOF:]
    full_ids = torch.cat((input_ids, action_ids.unsqueeze(0)), dim=1)
    v_cache.clear()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out2 = vla(input_ids=full_ids, pixel_values=inputs.pixel_values, output_attentions=True)
    A2 = out2.attentions[li][0].float()
    N2 = A2.shape[-1]
    vh2 = v_cache[li][0].float().view(N2, n_heads, head_dim)
    Lp = input_ids.shape[1]
    q_rows = [n_patch + Lp - 1 + k for k in range(N_DOF)]
    aw_vw, aw_w, aw_mass = per_head_maps(A2, vh2, Wo[li], q_rows, patch_cols)
    del gen, out2
    for h in handles:
        h.remove()
    del vla, processor
    torch.cuda.empty_cache()
    return dict(
        text_vw=tw_vw, text_w=tw_w, text_mass=tw_mass,
        act_vw=aw_vw, act_w=aw_w, act_mass=aw_mass,
        tokens=toks, prompt=prompt, rgb=img, n_heads=n_heads,
    )


def plot_head_grid(maps_c, maps_p, rgb_c, rgb_p, title, path, kind="mean_q"):
    """maps_*: (H, Q, 256). Show mean over queries as 16x16 overlay per head."""
    H = maps_c.shape[0]
    ncol, nrow = 8, (H + 7) // 8
    fig, axes = plt.subplots(nrow, ncol * 2, figsize=(2.0 * ncol * 2, 2.0 * nrow))
    for h in range(H):
        r, c = divmod(h, ncol)
        Mc = row_norm_h(maps_c[h:h+1])[0].mean(0)
        Mp = row_norm_h(maps_p[h:h+1])[0].mean(0)
        for j, (M, rgb, ax) in enumerate([
            (Mc, rgb_c, axes[r, 2 * c]),
            (Mp, rgb_p, axes[r, 2 * c + 1]),
        ]):
            up = zoom(M.reshape(GRID, GRID), rgb.shape[0] / GRID, order=1)
            ax.imshow(rgb); ax.imshow(up, cmap="jet", alpha=0.55)
            tag = "C" if j == 0 else "P"
            ax.set_title(f"h{h}/{tag} max={M.max():.2f}", fontsize=7)
            ax.axis("off")
    for k in range(H, nrow * ncol):
        r, c = divmod(k, ncol)
        axes[r, 2 * c].axis("off"); axes[r, 2 * c + 1].axis("off")
    fig.suptitle(title + "\n(each pair: CLEAN | BACKDOOR+TRIG)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def plot_head_stats(c, p, name, path):
    """Per-head entropy / f_norm / peak / mass for clean vs poison."""
    Hc = c["n_heads"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    x = np.arange(Hc)

    def stats(maps):
        rn = row_norm_h(maps)
        ent = np.array([entropy_rows(rn[h]) for h in range(Hc)])
        fn = np.array([fnorm_rows(rn[h]) for h in range(Hc)])
        peak = rn.max(axis=(1, 2))
        return ent, fn, peak

    for col, key, title in [(0, "text_vw", "TEXT→img α·‖Wov‖"), (1, "act_vw", "ACTION→img α·‖Wov‖")]:
        ent_c, fn_c, _ = stats(c[key])
        ent_p, fn_p, _ = stats(p[key])
        ax = axes[0, col]
        ax.bar(x - 0.2, ent_c, 0.4, label="clean", color="steelblue")
        ax.bar(x + 0.2, ent_p, 0.4, label="bd+trig", color="darkorange")
        ax.set_title(f"{title} entropy (↑ flatter)")
        ax.set_xlabel("head"); ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
        ax = axes[1, col]
        ax.bar(x - 0.2, fn_c, 0.4, label="clean", color="steelblue")
        ax.bar(x + 0.2, fn_p, 0.4, label="bd+trig", color="darkorange")
        ax.set_title(f"{title} f_norm (dispersion)")
        ax.set_xlabel("head"); ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"{name} per-head stats (last layer)")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)

    # mass importance
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, key, title in [
        (axes[0], "text_mass", "TEXT head mass Σ α·‖Wov‖"),
        (axes[1], "act_mass", "ACTION head mass Σ α·‖Wov‖"),
    ]:
        ax.bar(x - 0.2, c[key], 0.4, label="clean", color="steelblue")
        ax.bar(x + 0.2, p[key], 0.4, label="bd+trig", color="darkorange")
        ax.set_title(title); ax.set_xlabel("head"); ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"{name} which heads carry value-weighted mass")
    fig.tight_layout()
    fig.savefig(path.replace(".png", "_mass.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", path.replace(".png", "_mass.png"))


def plot_delta_ranking(c, p, path):
    """Rank heads by |Δentropy| and |Δf_norm| for text and action."""
    lines = ["Per-head clean vs bd+trig deltas (last layer, value-weighted)\n"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in [
        (axes[0], "text_vw", "TEXT→img"),
        (axes[1], "act_vw", "ACTION→img"),
    ]:
        rn_c, rn_p = row_norm_h(c[key]), row_norm_h(p[key])
        H = rn_c.shape[0]
        dent = np.array([entropy_rows(rn_p[h]) - entropy_rows(rn_c[h]) for h in range(H)])
        dfn = np.array([fnorm_rows(rn_p[h]) - fnorm_rows(rn_c[h]) for h in range(H)])
        order = np.argsort(-np.abs(dent))
        ax.bar(np.arange(H), dent[order], color=["crimson" if d > 0 else "steelblue" for d in dent[order]])
        ax.set_xticks(np.arange(H))
        ax.set_xticklabels([str(i) for i in order], fontsize=7)
        ax.set_xlabel("head (sorted by |Δentropy|)")
        ax.set_ylabel("entropy(bd) − entropy(clean)")
        ax.set_title(f"{title}: Δentropy (red=flatter under trigger)")
        ax.axhline(0, color="k", lw=0.8); ax.grid(True, axis="y", alpha=0.3)
        lines.append(f"\n=== {title} ===")
        lines.append(f"{'head':>4} {'Δent':>8} {'Δfnorm':>8} {'mass_c':>10} {'mass_p':>10}")
        mass_key = "text_mass" if "text" in key else "act_mass"
        for h in order:
            lines.append(f"{h:4d} {dent[h]:8.3f} {dfn[h]:8.4f} "
                         f"{c[mass_key][h]:10.3f} {p[mass_key][h]:10.3f}")
    fig.suptitle("GoBA per-head trigger effect (value-weighted)")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    with open(path.replace(".png", ".txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("saved", path)


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    os.makedirs(OUT, exist_ok=True)

    img_c, desc = get_scene("clean")
    img_p, _ = get_scene("poison")
    print(f"desc={desc!r}", flush=True)

    print("\n=== CLEAN ===", flush=True)
    clean = load_and_probe(CLEAN_CKPT, img_c, desc)
    print("\n=== GOBA BD ===", flush=True)
    poison = load_and_probe(GOBA_CKPT, img_p, desc)

    np.savez_compressed(
        f"{OUT}/perhead.npz",
        clean_text_vw=clean["text_vw"], poison_text_vw=poison["text_vw"],
        clean_text_w=clean["text_w"], poison_text_w=poison["text_w"],
        clean_act_vw=clean["act_vw"], poison_act_vw=poison["act_vw"],
        clean_act_w=clean["act_w"], poison_act_w=poison["act_w"],
        clean_text_mass=clean["text_mass"], poison_text_mass=poison["text_mass"],
        clean_act_mass=clean["act_mass"], poison_act_mass=poison["act_mass"],
        clean_rgb=clean["rgb"], poison_rgb=poison["rgb"],
        tokens=np.array(clean["tokens"], dtype=object),
        dof_names=np.array(DOF_NAMES, dtype=object),
        prompt=np.array(clean["prompt"]),
        note=np.array("per-head α_h * ||W_O^h v_h|| at last layer; t=0 after wait"),
    )

    plot_head_grid(clean["text_vw"], poison["text_vw"], clean["rgb"], poison["rgb"],
                   "GoBA TEXT→image per-head (α·‖Wov‖, mean over text tokens)",
                   f"{OUT}/GRID_text_perhead_overlay.png")
    plot_head_grid(clean["act_vw"], poison["act_vw"], clean["rgb"], poison["rgb"],
                   "GoBA ACTION→image per-head (α·‖Wov‖, mean over DoFs)",
                   f"{OUT}/GRID_action_perhead_overlay.png")
    plot_head_stats(clean, poison, "GoBA", f"{OUT}/STATS_perhead_entropy_fnorm.png")
    plot_delta_ranking(clean, poison, f"{OUT}/RANK_perhead_delta_entropy.png")

    # top-4 heads by |Δent| for text: full token×patch matrices clean|poison
    rn_c, rn_p = row_norm_h(clean["text_vw"]), row_norm_h(poison["text_vw"])
    dent = np.array([entropy_rows(rn_p[h]) - entropy_rows(rn_c[h]) for h in range(rn_c.shape[0])])
    top = np.argsort(-np.abs(dent))[:4]
    toks = clean["tokens"]
    L = len(toks)
    fig, axes = plt.subplots(4, 2, figsize=(11, 2.2 * 4))
    for i, h in enumerate(top):
        vmin = min(np.percentile(rn_c[h], 1), np.percentile(rn_p[h], 1))
        vmax = max(np.percentile(rn_c[h], 99), np.percentile(rn_p[h], 99))
        for j, (M, title) in enumerate([(rn_c[h], "CLEAN"), (rn_p[h], "BD+TRIG")]):
            ax = axes[i, j]
            im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_yticks(range(L)); ax.set_yticklabels(toks, fontsize=6)
            ax.set_title(f"head {h} {title}  Δent={dent[h]:+.3f}  max={M.max():.3f}", fontsize=9)
            ax.set_xlabel("patch")
        fig.colorbar(im, ax=axes[i].tolist(), shrink=0.8)
    fig.suptitle("GoBA TEXT→image: top-|Δentropy| heads (value-weighted, row-norm)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/TOP_text_heads_matrices.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved TOP_text_heads_matrices.png")

    rn_c, rn_p = row_norm_h(clean["act_vw"]), row_norm_h(poison["act_vw"])
    dent = np.array([entropy_rows(rn_p[h]) - entropy_rows(rn_c[h]) for h in range(rn_c.shape[0])])
    top = np.argsort(-np.abs(dent))[:4]
    fig, axes = plt.subplots(4, 2, figsize=(11, 2.0 * 4))
    for i, h in enumerate(top):
        vmin = min(np.percentile(rn_c[h], 1), np.percentile(rn_p[h], 1))
        vmax = max(np.percentile(rn_c[h], 99), np.percentile(rn_p[h], 99))
        for j, (M, title) in enumerate([(rn_c[h], "CLEAN"), (rn_p[h], "BD+TRIG")]):
            ax = axes[i, j]
            im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_yticks(range(N_DOF)); ax.set_yticklabels(DOF_NAMES, fontsize=7)
            ax.set_title(f"head {h} {title}  Δent={dent[h]:+.3f}  max={M.max():.3f}", fontsize=9)
        fig.colorbar(im, ax=axes[i].tolist(), shrink=0.8)
    fig.suptitle("GoBA ACTION→image: top-|Δentropy| heads (value-weighted, row-norm)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/TOP_action_heads_matrices.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved TOP_action_heads_matrices.png")

    with open(f"{OUT}/README.txt", "w") as f:
        f.write("Per-head value-weighted attention (α_h · ||W_O^h v_h||), last LLM layer.\n")
        f.write(f"Prompt: {clean['prompt']}\n")
        f.write(f"n_heads={clean['n_heads']}  tokens={clean['tokens']}\n")
        f.write("GRID_*: each head pair CLEAN|POISON overlay (mean over queries).\n")
        f.write("STATS_*: entropy & f_norm per head.\n")
        f.write("RANK_*: heads sorted by |Δentropy|.\n")
        f.write("TOP_*: matrices for the 4 most-changed heads.\n")
    print(f"\nDone -> {OUT}")


if __name__ == "__main__":
    main()
