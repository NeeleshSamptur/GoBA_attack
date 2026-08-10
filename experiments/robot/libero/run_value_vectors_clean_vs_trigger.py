"""
Compare LLM image-token value vectors: clean scene vs trigger scene.
Models: clean OpenVLA + GoBA-backdoored.
"""
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import json
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
GOBA_CKPT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
OUT = f"{REPO}/attn_maps/single_sample_analysis/value_vectors_clean_vs_trigger"
BDDL = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "trigger": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
SUITE = "libero_goal"
TASK_ID = 7
SEED = 7
NUM_STEPS_WAIT = 10
DEVICE = 0
N_DOF = 7
GRID = 16
LAYER = 31


def preprocess(img):
    im = tf.convert_to_tensor(np.array(Image.fromarray(img).convert("RGB")))
    dt = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, dt, saturate=True)
    return np.array(Image.fromarray(im.numpy()).convert("RGB"))


def get_scene(task, bddl, seed):
    env, desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False,
                               bddl_path=bddl, seed=seed)
    env.reset()
    obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
    img = preprocess(get_libero_image(obs, 224))
    env.close()
    return img, desc


def pca_rgb(X):
    pcs = PCA(n_components=3, random_state=0).fit_transform(X.astype(np.float64))
    for c in range(3):
        lo, hi = np.percentile(pcs[:, c], [2, 98])
        pcs[:, c] = np.clip((pcs[:, c] - lo) / (hi - lo + 1e-12), 0, 1)
    return pcs


def overlay(ax, img, flat, title, vmin=None, vmax=None, cmap="viridis"):
    up = zoom(flat.reshape(GRID, GRID), img.shape[0] / GRID, order=1)
    ax.imshow(img)
    kw = dict(cmap=cmap, alpha=0.55)
    if vmin is not None:
        kw["vmin"] = vmin
        kw["vmax"] = vmax
    im = ax.imshow(up, **kw)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    return im


def extract_V(vla, processor, img, desc):
    layers = vla.language_model.model.layers
    n_layers = len(layers)
    n_heads = layers[0].self_attn.num_heads
    head_dim = layers[0].self_attn.head_dim
    hidden = n_heads * head_dim
    n_patch = vla.vision_backbone.featurizer.patch_embed.num_patches
    v_cache = {}
    handles = [lyr.self_attn.v_proj.register_forward_hook(
        lambda _m, _i, out, l=l: v_cache.__setitem__(l, out))
        for l, lyr in enumerate(layers)]
    Wo = layers[LAYER].self_attn.o_proj.weight.detach().float().view(hidden, n_heads, head_dim)

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

    V = v_cache[LAYER][0].float()
    V_img = V[1:1 + n_patch].cpu().numpy()
    vh = V.view(V.shape[0], n_heads, head_dim)
    f = torch.einsum("nhd,ohd->nho", vh, Wo)
    wov = f.norm(dim=-1)[1:1 + n_patch].sum(-1).cpu().numpy()
    v_norm = np.linalg.norm(V_img, axis=1)
    for h in handles:
        h.remove()
    torch.cuda.empty_cache()
    return dict(V=V_img.astype(np.float32), v_norm=v_norm.astype(np.float32),
                wov_norm=wov.astype(np.float32), prompt=prompt, n_patch=n_patch)


def load_model(ckpt):
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE).eval()
    if os.path.isdir(ckpt) and os.path.exists(os.path.join(ckpt, "dataset_statistics.json")):
        vla.norm_stats = json.load(open(os.path.join(ckpt, "dataset_statistics.json")))
    return processor, vla


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    os.makedirs(OUT, exist_ok=True)

    task = benchmark.get_benchmark_dict()[SUITE]().get_task(TASK_ID)
    scenes = {}
    for role, bddl in BDDL.items():
        img, desc = get_scene(task, bddl, SEED)
        scenes[role] = (img, desc)
        Image.fromarray(img).save(f"{OUT}/rgb_{role}.png")
        print(f"{role}: {desc}", flush=True)

    # Shared prompt text from clean scene (same instruction)
    shared_desc = scenes["clean"][1]

    results = {}
    for model_name, ckpt in [("clean_model", CLEAN_CKPT), ("goba", GOBA_CKPT)]:
        print(f"\n=== {model_name} ===", flush=True)
        processor, vla = load_model(ckpt)
        for role, (img, _) in scenes.items():
            rec = extract_V(vla, processor, img, shared_desc)
            results[(model_name, role)] = rec
            print(f"  {role}: ‖V‖ mean={rec['v_norm'].mean():.2f} "
                  f"max@{rec['v_norm'].argmax()}={rec['v_norm'].max():.2f}  "
                  f"WoV max@{rec['wov_norm'].argmax()}={rec['wov_norm'].max():.1f}", flush=True)
        del vla, processor
        torch.cuda.empty_cache()

    # ---- Figure A: clean model clean vs trigger ----
    for model_name, label in [("clean_model", "Clean OpenVLA"), ("goba", "GoBA-backdoored")]:
        c = results[(model_name, "clean")]
        t = results[(model_name, "trigger")]
        img_c, img_t = scenes["clean"][0], scenes["trigger"][0]

        # align color scales for norms
        vn = np.concatenate([c["v_norm"], t["v_norm"]])
        wn = np.concatenate([c["wov_norm"], t["wov_norm"]])
        vmin_v, vmax_v = np.percentile(vn, [2, 98])
        vmin_w, vmax_w = np.percentile(wn, [2, 98])

        # cosine of corresponding patches (same index) — meaningful if layouts similar
        Vc = c["V"] / np.clip(np.linalg.norm(c["V"], axis=1, keepdims=True), 1e-12, None)
        Vt = t["V"] / np.clip(np.linalg.norm(t["V"], axis=1, keepdims=True), 1e-12, None)
        cos_diag = (Vc * Vt).sum(1)  # per-patch cos(clean, trigger)
        # delta norms
        d_v = t["v_norm"] - c["v_norm"]
        d_w = t["wov_norm"] - c["wov_norm"]
        # L2 distance between value vectors
        dist = np.linalg.norm(t["V"] - c["V"], axis=1)

        fig, axes = plt.subplots(3, 4, figsize=(15, 11))
        # row0: RGB
        axes[0, 0].imshow(img_c); axes[0, 0].set_title("clean scene RGB"); axes[0, 0].axis("off")
        axes[0, 1].imshow(img_t); axes[0, 1].set_title("trigger scene RGB"); axes[0, 1].axis("off")
        axes[0, 2].axis("off"); axes[0, 3].axis("off")
        # annotate trigger presence
        axes[0, 2].text(0.05, 0.6, f"{label}\nL{LAYER} image-token V\n"
                         f"mean cos(V_c,V_t)={cos_diag.mean():.3f}\n"
                         f"mean ‖V_t−V_c‖={dist.mean():.2f}",
                         transform=axes[0, 2].transAxes, fontsize=11, va="center")

        # row1: ||V||
        im = overlay(axes[1, 0], img_c, c["v_norm"], "clean ‖V‖₂", vmin_v, vmax_v)
        overlay(axes[1, 1], img_t, t["v_norm"], "trigger ‖V‖₂", vmin_v, vmax_v)
        fig.colorbar(im, ax=axes[1, :2].tolist(), fraction=0.03, pad=0.02)
        lim = np.percentile(np.abs(d_v), 98)
        imd = overlay(axes[1, 2], img_t, d_v, "Δ‖V‖ (trig−clean)", -lim, lim, cmap="RdBu_r")
        fig.colorbar(imd, ax=axes[1, 2], fraction=0.046)
        imc = overlay(axes[1, 3], img_t, cos_diag, "per-patch cos(V_c,V_t)", 0.5, 1.0, cmap="magma")
        fig.colorbar(imc, ax=axes[1, 3], fraction=0.046)

        # row2: WoV + PCA
        im = overlay(axes[2, 0], img_c, c["wov_norm"], "clean ∑‖Wₒv‖", vmin_w, vmax_w)
        overlay(axes[2, 1], img_t, t["wov_norm"], "trigger ∑‖Wₒv‖", vmin_w, vmax_w)
        fig.colorbar(im, ax=axes[2, :2].tolist(), fraction=0.03, pad=0.02)
        # joint PCA for fair colors
        both = np.concatenate([c["V"], t["V"]], 0)
        rgb_both = pca_rgb(both)
        rgb_c = rgb_both[:256].reshape(GRID, GRID, 3)
        rgb_t = rgb_both[256:].reshape(GRID, GRID, 3)
        for ax, rgb, im0, title in (
            (axes[2, 2], rgb_c, img_c, "clean PCA(V)"),
            (axes[2, 3], rgb_t, img_t, "trigger PCA(V)"),
        ):
            up = zoom(rgb, (im0.shape[0] / GRID, im0.shape[0] / GRID, 1), order=1)
            ax.imshow(im0); ax.imshow(up, alpha=0.65)
            ax.set_title(title, fontsize=9); ax.axis("off")

        fig.suptitle(f"{label} · value vectors · clean vs trigger · task={TASK_ID} seed={SEED}",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(f"{OUT}/FIGURE_{model_name}_clean_vs_trigger.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved FIGURE_{model_name}_clean_vs_trigger.png  "
              f"mean_cos={cos_diag.mean():.3f} mean_dist={dist.mean():.2f}", flush=True)

    # ---- Figure B: 2x2 ‖V‖ comparison all four conditions ----
    order = [("clean_model", "clean"), ("clean_model", "trigger"),
             ("goba", "clean"), ("goba", "trigger")]
    labels = ["Clean model × clean", "Clean model × TRIGGER",
              "GoBA × clean", "GoBA × TRIGGER"]
    alln = np.concatenate([results[k]["v_norm"] for k in order])
    vmin, vmax = np.percentile(alln, [2, 98])
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for ax, key, lab in zip(axes.ravel(), order, labels):
        role = key[1]
        overlay(ax, scenes[role][0], results[key]["v_norm"], lab, vmin, vmax)
    fig.suptitle(f"L{LAYER} ‖V‖₂ on image tokens · shared color scale", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/FIGURE_2x2_Vnorm.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure C: GoBA trigger − clean_model trigger (same scene, different weights) ----
    # and GoBA trigger − GoBA clean
    pairs = [
        (("goba", "trigger"), ("clean_model", "trigger"), "GoBA−CleanModel on TRIGGER scene",
         scenes["trigger"][0]),
        (("goba", "trigger"), ("goba", "clean"), "GoBA: TRIGGER−clean scene",
         scenes["trigger"][0]),
        (("clean_model", "trigger"), ("clean_model", "clean"), "CleanModel: TRIGGER−clean scene",
         scenes["trigger"][0]),
    ]
    fig, axes = plt.subplots(len(pairs), 3, figsize=(12, 3.8 * len(pairs)))
    for r, ((a, b), (c0, c1), title, img) in enumerate(pairs):
        Va, Vb = results[(a, b)]["V"], results[(c0, c1)]["V"]
        # if different scenes, still compare by index
        dnorm = np.linalg.norm(Va - Vb, axis=1)
        na = results[(a, b)]["v_norm"]
        nb = results[(c0, c1)]["v_norm"]
        Va_n = Va / np.clip(np.linalg.norm(Va, axis=1, keepdims=True), 1e-12, None)
        Vb_n = Vb / np.clip(np.linalg.norm(Vb, axis=1, keepdims=True), 1e-12, None)
        cos = (Va_n * Vb_n).sum(1)
        lim = np.percentile(dnorm, 98)
        im0 = overlay(axes[r, 0], img, dnorm, f"{title}\n‖ΔV‖₂", 0, lim, cmap="magma")
        lim2 = np.percentile(np.abs(na - nb), 98)
        im1 = overlay(axes[r, 1], img, na - nb, "Δ‖V‖", -lim2, lim2, cmap="RdBu_r")
        im2 = overlay(axes[r, 2], img, cos, "cos(V_a,V_b)", 0.3, 1.0, cmap="magma")
        for im, ax in zip((im0, im1, im2), axes[r]):
            fig.colorbar(im, ax=ax, fraction=0.046)
        print(f"{title}: mean‖ΔV‖={dnorm.mean():.2f} mean_cos={cos.mean():.3f}", flush=True)
    fig.suptitle("Value-vector differences (image tokens)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/FIGURE_value_deltas.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        f"{OUT}/compare.npz",
        **{f"{m}_{r}_V": results[(m, r)]["V"] for m, r in order},
        **{f"{m}_{r}_v_norm": results[(m, r)]["v_norm"] for m, r in order},
        **{f"{m}_{r}_wov": results[(m, r)]["wov_norm"] for m, r in order},
        rgb_clean=scenes["clean"][0], rgb_trigger=scenes["trigger"][0],
        task_id=TASK_ID, seed=SEED, layer=LAYER,
    )
    print(f"Done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
