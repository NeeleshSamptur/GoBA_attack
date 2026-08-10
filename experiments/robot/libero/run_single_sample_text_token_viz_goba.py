"""
Per TEXT-TOKEN -> image attention overlays for the GoBA single-sample setup.

Same (task=7, seed=7, prompt) as run_single_sample_attention_viz_goba.py, but
now each TEXT token in the prompt gets its OWN attention map over the 256 image
patches (last LLM layer, head-averaged) -- this is the T2IShield-style
text->image view, not the action-DoF maps from the previous run.

Conditions:
  clean_model / goba_backdoored  x  clean_scene / trigger_scene

Output: attn_maps/single_sample_analysis/goba_text_tokens/
"""
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from libero.libero import benchmark
from PIL import Image
from scipy.ndimage import zoom
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.libero.run_libero_eval_attentionmap import (
    crop_and_resize,
    get_avg_patch_text_attention,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer

GOBA_CKPT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
CLEAN_CKPT = "openvla/openvla-7b-finetuned-libero-goal"
OUT_DIR = f"{REPO}/attn_maps/single_sample_analysis/goba_text_tokens"

BDDL = {
    "clean_scene": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "trigger_scene": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
SUITE = "libero_goal"
TASK_ID = 7
SEED = 7
NUM_STEPS_WAIT = 10
DEVICE = 0
GRID = 16
LAYER = -1   # last layer (same as T2IShield / assimilation detector)


def preprocess(img):
    im = tf.convert_to_tensor(np.array(Image.fromarray(img).convert("RGB")))
    dt = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, dt, saturate=True)
    return np.array(Image.fromarray(im.numpy()).convert("RGB"))


def get_scene(task, bddl_dir, seed):
    env, desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False,
                               bddl_path=bddl_dir, seed=seed)
    env.reset()
    obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
    img = preprocess(get_libero_image(obs, 224))
    env.close()
    return img, desc


def load_model(ckpt):
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE)
    if os.path.isdir(ckpt) and os.path.exists(os.path.join(ckpt, "dataset_statistics.json")):
        vla.norm_stats = json.load(open(os.path.join(ckpt, "dataset_statistics.json")))
    return processor, vla


def safe_tok_label(tok: str, idx: int) -> str:
    s = tok.replace("\n", "\\n").replace("▁", " ").replace("Ġ", " ")
    s = re.sub(r"[^\w\\\.\-\+\*=\?\!:,;/\(\) ]+", "", s).strip() or f"tok{idx}"
    return f"{idx}:{s[:18]}"


def text_token_maps(vla, processor, action_tokenizer, img, desc):
    """Return row-normalized (n_text, 256) maps + token strings + f_norm."""
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, Image.fromarray(img)).to(DEVICE, dtype=torch.bfloat16)
    # Teacher-forced prompt forward (same as assimilation / T2IShield scripts).
    # Do NOT append 29871 here -- that token is only for generate()/predict_action
    # and breaks text_mask vs attention-seq alignment in get_avg_patch_text_attention.

    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(**inputs, output_attentions=True)

    mask = (inputs.input_ids < action_tokenizer.action_token_begin_idx)[0]
    _, text2patch = get_avg_patch_text_attention(out.attentions, num_patches, mask, layer=LAYER)
    maps = text2patch.float().cpu().numpy()                         # (n_text, 256)
    maps = maps / np.clip(maps.sum(axis=1, keepdims=True), 1e-12, None)

    # token strings for the rows that survived the mask (mask[1:] indexing inside helper)
    ids = inputs.input_ids[0].tolist()
    kept = []
    for i, tid in enumerate(ids):
        if i == 0:
            continue  # dropped by text_mask[1:]
        if bool(mask[i]):
            kept.append(tid)
    tok_strs = processor.tokenizer.convert_ids_to_tokens(kept)
    # length guard
    n = min(len(tok_strs), maps.shape[0])
    tok_strs, maps = tok_strs[:n], maps[:n]

    mbar = maps.mean(0)
    f_norm = float(np.linalg.norm(maps - mbar[None, :], axis=1).mean())
    del out
    torch.cuda.empty_cache()
    return maps, tok_strs, prompt, f_norm


def overlay_all_tokens(img, maps, tok_strs, out_path, suptitle, ncols=5):
    n = len(maps)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.8 * nrows))
    axes = np.atleast_2d(axes)
    allv = np.concatenate([m.flatten() for m in maps])
    vmin, vmax = np.percentile(allv, 1), np.percentile(allv, 99)
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i >= n:
            ax.axis("off")
            continue
        up = zoom(maps[i].reshape(GRID, GRID), img.shape[0] / GRID, order=1)
        ax.imshow(img)
        im = ax.imshow(up, cmap="jet", alpha=0.55, vmin=vmin, vmax=vmax)
        ax.set_title(f"{safe_tok_label(tok_strs[i], i)}\nmax={maps[i].max():.3f}", fontsize=7)
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, label="text→patch attn")
    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    os.makedirs(OUT_DIR, exist_ok=True)
    task = benchmark.get_benchmark_dict()[SUITE]().get_task(TASK_ID)

    scenes = {}
    for role, bddl in BDDL.items():
        img, desc = get_scene(task, bddl, SEED)
        scenes[role] = (img, desc)
        Image.fromarray(img).save(f"{OUT_DIR}/rgb_{role}_t{TASK_ID}_s{SEED}.png")

    shared_desc = scenes["clean_scene"][1]
    prompt_str = f"In: What action should the robot take to {shared_desc.lower()}?\nOut:"
    with open(f"{OUT_DIR}/PROMPT.txt", "w") as f:
        f.write(prompt_str + "\n")
        f.write(f"task_id={TASK_ID} seed={SEED} layer={LAYER} (last)\n")
        f.write("EACH panel = ONE text token's attention over the full image.\n")
    print(f"SHARED PROMPT:\n{prompt_str}", flush=True)

    results = {}
    for model_name, ckpt in [("clean_model", CLEAN_CKPT), ("goba_backdoored", GOBA_CKPT)]:
        print(f"\n=== Loading {model_name} ===", flush=True)
        processor, vla = load_model(ckpt)
        action_tokenizer = ActionTokenizer(processor.tokenizer)
        for role, (img, _) in scenes.items():
            maps, toks, prompt, fnorm = text_token_maps(
                vla, processor, action_tokenizer, img, shared_desc)
            results[(model_name, role)] = (maps, toks, fnorm)
            overlay_all_tokens(
                img, maps, toks,
                f"{OUT_DIR}/{model_name}__{role}__ALL_TEXT_TOKENS.png",
                f"GoBA TEXT→image | {model_name} | {role} | last layer | f_norm={fnorm:.4f}\n{prompt.strip()}",
            )
            # also save each token as its own file for close inspection
            tok_dir = f"{OUT_DIR}/{model_name}__{role}__per_token"
            os.makedirs(tok_dir, exist_ok=True)
            for i, (m, t) in enumerate(zip(maps, toks)):
                fig, ax = plt.subplots(figsize=(4, 4))
                up = zoom(m.reshape(GRID, GRID), img.shape[0] / GRID, order=1)
                ax.imshow(img)
                im = ax.imshow(up, cmap="jet", alpha=0.55)
                ax.set_title(f"{safe_tok_label(t, i)}  max={m.max():.3f} argmax={m.argmax()}", fontsize=9)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046)
                fig.savefig(f"{tok_dir}/tok_{i:02d}.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
            print(f"  {model_name}/{role}: n_text={len(toks)} f_norm={fnorm:.4f} "
                  f"mean_max={maps.max(1).mean():.4f} tokens={toks}", flush=True)
        del vla, processor
        torch.cuda.empty_cache()

    # pooled (mean over text tokens) 2x2 summary
    order = [("clean_model", "clean_scene"), ("clean_model", "trigger_scene"),
             ("goba_backdoored", "clean_scene"), ("goba_backdoored", "trigger_scene")]
    pooled = [results[k][0].mean(0) for k in order]
    allv = np.concatenate([p.flatten() for p in pooled])
    vmin, vmax = np.percentile(allv, 1), np.percentile(allv, 99)
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    labels = ["CLEAN model × clean scene", "CLEAN model × TRIGGER scene",
              "GoBA-backdoored × clean scene", "GoBA-backdoored × TRIGGER scene"]
    for ax, (mn, role), p, lab in zip(axes.ravel(), order, pooled, labels):
        img = scenes[role][0]
        fn = results[(mn, role)][2]
        up = zoom(p.reshape(GRID, GRID), img.shape[0] / GRID, order=1)
        ax.imshow(img)
        im = ax.imshow(up, cmap="jet", alpha=0.55, vmin=vmin, vmax=vmax)
        ax.set_title(f"{lab}\nmax={p.max():.3f} f_norm={fn:.4f}", fontsize=10)
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="mean text→patch attn")
    fig.suptitle(f"GoBA TEXT-token pooled attention (last layer)\n{prompt_str.strip()}", fontsize=11)
    fig.savefig(f"{OUT_DIR}/SUMMARY_2x2_text_pooled.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez(f"{OUT_DIR}/maps.npz",
             **{f"{mn}__{role}": results[(mn, role)][0] for mn, role in order},
             tokens_clean_model_clean=np.array(results[("clean_model", "clean_scene")][1], dtype=object),
             f_norm=np.array([results[k][2] for k in order]),
             order=np.array([f"{a}__{b}" for a, b in order]),
             task_id=TASK_ID, seed=SEED, layer=np.array([LAYER]))
    print(f"\nDone. Per-text-token maps -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
