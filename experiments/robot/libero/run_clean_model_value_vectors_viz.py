"""
Clean OpenVLA: visualize LLM value vectors for all image patch tokens.
First pass, clean scene — V and W_O V (per-token), last layer (+ optional mid).
"""
import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from libero.libero import benchmark
from PIL import Image
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.libero.run_libero_eval_attentionmap import crop_and_resize
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

CLEAN_CKPT = "openvla/openvla-7b-finetuned-libero-goal"
OUT = f"{REPO}/attn_maps/single_sample_analysis/clean_model_value_vectors"
SUITE = "libero_goal"
TASK_ID = 7
SEED = 7
NUM_STEPS_WAIT = 10
DEVICE = 0
N_DOF = 7
GRID = 16
# show last layer + a mid layer for context
LAYERS_SHOW = [15, 24, 31]


def preprocess(img):
    im = tf.convert_to_tensor(np.array(Image.fromarray(img).convert("RGB")))
    dt = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, dt, saturate=True)
    return np.array(Image.fromarray(im.numpy()).convert("RGB"))


def overlay(ax, img, flat, title, cmap="viridis"):
    up = zoom(flat.reshape(GRID, GRID), img.shape[0] / GRID, order=1)
    ax.imshow(img)
    im = ax.imshow(up, cmap=cmap, alpha=0.55)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return im


def pca_rgb(X, n=3):
    """X: (N, D) -> (N, 3) in [0,1] via PCA."""
    pcs = PCA(n_components=n, random_state=0).fit_transform(X.astype(np.float64))
    for c in range(n):
        lo, hi = np.percentile(pcs[:, c], [2, 98])
        pcs[:, c] = np.clip((pcs[:, c] - lo) / (hi - lo + 1e-12), 0, 1)
    return pcs


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    os.makedirs(OUT, exist_ok=True)

    task = benchmark.get_benchmark_dict()[SUITE]().get_task(TASK_ID)
    bddl = f"{REPO}/BadLIBERO/libero/libero/bddl_files"
    env, desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False,
                               bddl_path=bddl, seed=SEED)
    env.reset()
    obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
    img = preprocess(get_libero_image(obs, 224))
    env.close()
    Image.fromarray(img).save(f"{OUT}/rgb.png")

    print("Loading clean model...", flush=True)
    processor = AutoProcessor.from_pretrained(CLEAN_CKPT, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        CLEAN_CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE).eval()
    n_patch = vla.vision_backbone.featurizer.patch_embed.num_patches

    layers = vla.language_model.model.layers
    n_layers = len(layers)
    n_heads = layers[0].self_attn.num_heads
    head_dim = layers[0].self_attn.head_dim
    hidden = n_heads * head_dim
    v_cache = {}
    handles = [lyr.self_attn.v_proj.register_forward_hook(
        lambda _m, _i, out, l=l: v_cache.__setitem__(l, out))
        for l, lyr in enumerate(layers)]
    Wo = {l: layers[l].self_attn.o_proj.weight.detach().float().view(hidden, n_heads, head_dim)
          for l in range(n_layers)}

    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, Image.fromarray(img)).to(DEVICE, dtype=torch.bfloat16)
    input_ids = inputs.input_ids
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        gen = vla.generate(
            input_ids, pixel_values=inputs.pixel_values,
            max_new_tokens=N_DOF, do_sample=False, return_dict_in_generate=True,
            output_attentions=False,
        )
    seq = gen.sequences
    del gen
    v_cache.clear()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _ = vla(input_ids=seq, pixel_values=inputs.pixel_values, return_dict=True)

    # Image tokens sit after BOS: indices [1, 1+n_patch)
    img_slice = slice(1, 1 + n_patch)
    save = {}

    # ---- Figure 1: ||V|| and ||W_O V|| spatial maps across layers ----
    nL = len(LAYERS_SHOW)
    fig, axes = plt.subplots(nL, 4, figsize=(14, 3.4 * nL))
    for r, l in enumerate(LAYERS_SHOW):
        V = v_cache[l][0].float()  # (N, hidden)
        V_img = V[img_slice].cpu().numpy()  # (256, H)
        # per-head V: (N, heads, d) then W_O v as Kobayashi f_norms
        vh = V.view(V.shape[0], n_heads, head_dim)
        f = torch.einsum("nhd,ohd->nho", vh, Wo[l])  # (N, heads, hidden_out_per? wait o is hidden)
        # Wo: (hidden, heads, head_dim); f: (N, heads, hidden) — actually o_proj is (hidden, hidden)
        # same as qktv scripts: f_norms = ||W_O^h v_h|| over output dim
        f_norms = f.norm(dim=-1)  # (N, heads)
        wov_norm = f_norms[img_slice].sum(-1).cpu().numpy()  # sum over heads
        v_norm = np.linalg.norm(V_img, axis=1)
        # mean per-head ||v_h||
        vh_img = vh[img_slice].cpu().numpy()
        v_head_norm = np.linalg.norm(vh_img, axis=-1).mean(-1)

        save[f"L{l}_V"] = V_img.astype(np.float32)
        save[f"L{l}_V_norm"] = v_norm.astype(np.float32)
        save[f"L{l}_WoV_norm"] = wov_norm.astype(np.float32)

        im0 = overlay(axes[r, 0], img, v_norm, f"L{l}  ‖V‖₂")
        im1 = overlay(axes[r, 1], img, v_head_norm, f"L{l}  mean_h ‖v_h‖")
        im2 = overlay(axes[r, 2], img, wov_norm, f"L{l}  ∑_h ‖Wₒʰ v_h‖")
        # PCA of V as RGB in patch grid
        rgb = pca_rgb(V_img).reshape(GRID, GRID, 3)
        up = zoom(rgb, (img.shape[0] / GRID, img.shape[0] / GRID, 1), order=1)
        axes[r, 3].imshow(img)
        axes[r, 3].imshow(up, alpha=0.65)
        axes[r, 3].set_title(f"L{l}  PCA(V)→RGB", fontsize=10)
        axes[r, 3].axis("off")
        for im, c in zip((im0, im1, im2), range(3)):
            fig.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.02)
        print(f"L{l}: V_img={V_img.shape}  ‖V‖ mean={v_norm.mean():.3f} "
              f"max@{v_norm.argmax()}={v_norm.max():.3f}  "
              f"WoV max@{wov_norm.argmax()}={wov_norm.max():.3f}", flush=True)

    fig.suptitle(
        f"Clean OpenVLA · LLM value vectors on image tokens only\n"
        f"first pass · task={TASK_ID} seed={SEED} · {prompt.strip()}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/FIGURE_value_norms_and_pca.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: last-layer only, larger — V norms + PCA + pairwise cosine ----
    l = 31
    V_img = save[f"L{l}_V"]
    v_norm = save[f"L{l}_V_norm"]
    wov_norm = save[f"L{l}_WoV_norm"]
    # cosine similarity among patches (optional subsample for readability: full 256x256 is fine)
    X = V_img / np.clip(np.linalg.norm(V_img, axis=1, keepdims=True), 1e-12, None)
    cos = X @ X.T

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    overlay(axes[0], img, v_norm, "L31 ‖V‖₂ (image tokens)")
    overlay(axes[1], img, wov_norm, "L31 ∑_h ‖Wₒʰ v_h‖")
    rgb = pca_rgb(V_img).reshape(GRID, GRID, 3)
    up = zoom(rgb, (img.shape[0] / GRID, img.shape[0] / GRID, 1), order=1)
    axes[2].imshow(img)
    axes[2].imshow(up, alpha=0.65)
    axes[2].set_title("L31 PCA(V) → RGB")
    axes[2].axis("off")
    im = axes[3].imshow(cos, cmap="coolwarm", vmin=-0.2, vmax=1.0)
    axes[3].set_title("L31 patch×patch cos(V)")
    axes[3].set_xlabel("patch j"); axes[3].set_ylabel("patch i")
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    fig.suptitle("Value vectors alone · image tokens · last LLM layer", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/FIGURE_L31_values_detail.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # bare value-norm map without RGB (pure ‖V‖ grid)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, flat, title in zip(
        axes,
        [v_norm, wov_norm, np.linalg.norm(V_img - V_img.mean(0), axis=1)],
        ["‖V‖₂", "∑_h ‖Wₒʰ v_h‖", "‖V − V̄‖₂"],
    ):
        im = ax.imshow(flat.reshape(GRID, GRID), cmap="magma")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("L31 image-token value magnitudes (16×16 grid, no RGB)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/FIGURE_L31_value_grids.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    for h in handles:
        h.remove()
    np.savez_compressed(
        f"{OUT}/value_vectors.npz",
        rgb=img, prompt=np.array(prompt), task_id=TASK_ID, seed=SEED,
        n_patch=n_patch, layers=np.array(LAYERS_SHOW),
        **save,
        L31_cosine=cos.astype(np.float32),
    )
    print(f"Saved figures -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
