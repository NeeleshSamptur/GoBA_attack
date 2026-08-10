"""
Single-sample attention overlay analysis for GoBA.

For ONE fixed (task, seed, prompt):
  - Clean OpenVLA (official libero_goal fine-tune)
  - GoBA backdoored OpenVLA
each run on:
  - clean BDDL scene (no trigger object)
  - poison BDDL scene (physical trigger object present)

Saves, for every action DoF token, the compression-band attention map
overlaid on the full RGB image, plus a pooled summary 2x2 panel.

Output: attn_maps/single_sample_analysis/goba/
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
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.libero.run_libero_eval_attentionmap import crop_and_resize
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

GOBA_CKPT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
CLEAN_CKPT = "openvla/openvla-7b-finetuned-libero-goal"
OUT_DIR = f"{REPO}/attn_maps/single_sample_analysis/goba"

BDDL = {
    "clean_scene": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "trigger_scene": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
SUITE = "libero_goal"
TASK_ID = 7          # put_the_bowl_on_the_plate (familiar)
SEED = 7             # same seed -> closest paired layouts
NUM_STEPS_WAIT = 10
DEVICE = 0
N_DOF = 7
DOF_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
BAND = [8, 16, 24, 27]
GRID = 16


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


def action_dof_maps(vla, processor, img, desc):
    """Return (7, 256) band-avg action-query maps + (7,) continuous action + prompt."""
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, Image.fromarray(img)).to(DEVICE, dtype=torch.bfloat16)
    input_ids = inputs.input_ids
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        gen = vla.generate(input_ids, pixel_values=inputs.pixel_values,
                           max_new_tokens=N_DOF, output_attentions=True,
                           return_dict_in_generate=True, do_sample=False)
    maps = np.zeros((N_DOF, num_patches), np.float32)
    for k in range(N_DOF):
        m = np.zeros(num_patches, np.float64)
        for l in BAND:
            la = gen.attentions[k][l][0].float().mean(dim=0)
            row = (la[-1] if la.dim() == 2 else la[0]).cpu().numpy()
            r = row[1: 1 + num_patches]
            m += r / max(r.sum(), 1e-12)
        maps[k] = m / len(BAND)
    tok = gen.sequences[0, -N_DOF:].cpu().numpy()
    del gen
    torch.cuda.empty_cache()
    return maps, prompt, tok


def overlay_grid(img, maps, titles, out_path, suptitle, ncols=4):
    n = len(maps)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
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
        ax.set_title(titles[i], fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="attn mass")
    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    os.makedirs(OUT_DIR, exist_ok=True)
    task_suite = benchmark.get_benchmark_dict()[SUITE]()
    task = task_suite.get_task(TASK_ID)

    # --- capture both scenes once (same seed) ---
    scenes = {}
    for role, bddl in BDDL.items():
        img, desc = get_scene(task, bddl, SEED)
        scenes[role] = (img, desc)
        Image.fromarray(img).save(f"{OUT_DIR}/rgb_{role}_t{TASK_ID}_s{SEED}.png")
        print(f"scene={role} desc='{desc}' img={img.shape}", flush=True)

    # Force the SAME prompt text for both scenes (user asked for serious/same prompt).
    # Use the clean-scene language instruction as the shared prompt body.
    shared_desc = scenes["clean_scene"][1]
    prompt_str = f"In: What action should the robot take to {shared_desc.lower()}?\nOut:"
    with open(f"{OUT_DIR}/PROMPT.txt", "w") as f:
        f.write(prompt_str + "\n")
        f.write(f"task_id={TASK_ID} seed={SEED}\n")
        f.write(f"clean_scene_desc={scenes['clean_scene'][1]}\n")
        f.write(f"trigger_scene_desc={scenes['trigger_scene'][1]}\n")
    print(f"SHARED PROMPT:\n{prompt_str}", flush=True)

    results = {}  # (model_name, scene_role) -> maps
    for model_name, ckpt in [("clean_model", CLEAN_CKPT), ("goba_backdoored", GOBA_CKPT)]:
        print(f"\n=== Loading {model_name} ===", flush=True)
        processor, vla = load_model(ckpt)
        for role, (img, _) in scenes.items():
            maps, prompt, tok = action_dof_maps(vla, processor, img, shared_desc)
            results[(model_name, role)] = maps
            # per-model per-scene: all 7 DoFs
            titles = [f"{DOF_NAMES[k]}  max={maps[k].max():.3f}  argmax={maps[k].argmax()}"
                      for k in range(N_DOF)]
            titles.append(f"POOLED  max={maps.mean(0).max():.3f}")
            overlay_grid(
                img, list(maps) + [maps.mean(0)], titles,
                f"{OUT_DIR}/{model_name}__{role}__per_dof.png",
                f"GoBA | {model_name} | {role} | band={BAND}\n{prompt_str.strip()}",
            )
            print(f"  {model_name}/{role}: tokens={tok.tolist()} "
                  f"pooled_argmax={maps.mean(0).argmax()} "
                  f"pooled_max={maps.mean(0).max():.4f}", flush=True)
        del vla, processor
        torch.cuda.empty_cache()

    # --- summary 2x2 pooled maps (shared color scale) ---
    order = [("clean_model", "clean_scene"), ("clean_model", "trigger_scene"),
             ("goba_backdoored", "clean_scene"), ("goba_backdoored", "trigger_scene")]
    pooled = [results[k].mean(0) for k in order]
    allv = np.concatenate([p.flatten() for p in pooled])
    vmin, vmax = np.percentile(allv, 1), np.percentile(allv, 99)
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    labels = ["CLEAN model × clean scene", "CLEAN model × TRIGGER scene",
              "GoBA-backdoored × clean scene", "GoBA-backdoored × TRIGGER scene"]
    for ax, (mn, role), p, lab in zip(axes.ravel(), order, pooled, labels):
        img = scenes[role][0]
        up = zoom(p.reshape(GRID, GRID), img.shape[0] / GRID, order=1)
        ax.imshow(img)
        im = ax.imshow(up, cmap="jet", alpha=0.55, vmin=vmin, vmax=vmax)
        ax.set_title(f"{lab}\nmax={p.max():.3f} argmax={p.argmax()}", fontsize=10)
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="pooled action-query attn")
    fig.suptitle(f"GoBA single-sample pooled attention\n{prompt_str.strip()}", fontsize=11)
    fig.savefig(f"{OUT_DIR}/SUMMARY_2x2_pooled.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- per-DoF comparison strips (one figure per DoF across 4 conditions) ---
    for d in range(N_DOF):
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
        vals = [results[k][d] for k in order]
        allv = np.concatenate([v.flatten() for v in vals])
        vmin, vmax = np.percentile(allv, 1), np.percentile(allv, 99)
        for ax, (mn, role), m, lab in zip(axes, order, vals, labels):
            img = scenes[role][0]
            up = zoom(m.reshape(GRID, GRID), img.shape[0] / GRID, order=1)
            ax.imshow(img)
            im = ax.imshow(up, cmap="jet", alpha=0.55, vmin=vmin, vmax=vmax)
            ax.set_title(f"{lab}\nmax={m.max():.3f}", fontsize=8)
            ax.axis("off")
        fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
        fig.suptitle(f"GoBA DoF={DOF_NAMES[d]}  |  {prompt_str.strip()}", fontsize=10)
        fig.savefig(f"{OUT_DIR}/COMPARE_dof_{d}_{DOF_NAMES[d]}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    np.savez(f"{OUT_DIR}/maps.npz",
             **{f"{mn}__{role}": results[(mn, role)] for mn, role in order},
             band=np.array(BAND), task_id=TASK_ID, seed=SEED,
             prompt=np.array(prompt_str))
    print(f"\nDone. All figures -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
