"""
GoBA companion to BadVLA's run_attention_assimilation_heatmap.py -- same experiment,
run on vanilla OpenVLA (GoBA) instead of openvla-oft (BadVLA), for a direct comparison.

Difference from BadVLA's version: GoBA is autoregressive, so there is a REAL action
token to query from (no placeholder stand-in needed). We run actual generation
(model.generate(..., output_attentions=True, return_dict_in_generate=True)) and
capture attention from each of the 7 generated action tokens back to the image
patches, at the last layer -- the genuine analog of BadVLA's action-query stat,
not an approximation.

Text-token attention reuses the already-validated extraction from
run_attention_assimilation_detector.py (f_norm AUROC=0.993 on this same repo).

Held-out tasks (7,8,9), 5 seeds, clean vs the real trigger box, matching BadVLA's
protocol exactly for apples-to-apples comparison.

Outputs:
  - attn_maps/goba_assimilation_stats.npz
  - attn_maps/goba_heatmaps/READABLE_*.png  (percentile-clipped, side-by-side)
"""
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

CHECKPOINT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
OUT_NPZ = f"{REPO}/attn_maps/goba_assimilation_stats.npz"
HEATMAP_DIR = f"{REPO}/attn_maps/goba_heatmaps"

CONDITIONS = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
SUITE = "libero_goal"
HELDOUT_TASKS = [7, 8, 9]
SEEDS = [7, 42, 1234, 2026, 31337]
NUM_STEPS_WAIT = 10
GRID = 16
DEVICE = 0
N_SAVE_EXAMPLES = 3

STAT_NAMES = ["f_norm", "mean_pairwise_cos", "consensus_entropy", "mean_token_entropy", "mean_token_max"]


def compute_stats(t2p):
    p = t2p / np.clip(t2p.sum(axis=1, keepdims=True), 1e-8, None)
    L, N = p.shape
    mbar = p.mean(axis=0)
    f_norm = np.linalg.norm(p - mbar[None, :], axis=1).mean()
    pn = p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-8, None)
    sim = pn @ pn.T
    iu = np.triu_indices(L, k=1)
    mean_pairwise_cos = sim[iu].mean() if L > 1 else float("nan")
    logN = np.log(N)
    consensus_entropy = -np.sum(mbar * np.log(mbar + 1e-12)) / logN
    mean_token_entropy = (-np.sum(p * np.log(p + 1e-12), axis=1) / logN).mean()
    mean_token_max = p.max(axis=1).mean()
    return {
        "f_norm": float(f_norm), "mean_pairwise_cos": float(mean_pairwise_cos),
        "consensus_entropy": float(consensus_entropy), "mean_token_entropy": float(mean_token_entropy),
        "mean_token_max": float(mean_token_max),
    }


def get_scene_image(task, cond_dir, seed):
    env, desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False, bddl_path=cond_dir, seed=seed)
    env.reset()
    obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
    img = get_libero_image(obs, 224)
    env.close()
    image = Image.fromarray(img).convert("RGB")
    im = tf.convert_to_tensor(np.array(image))
    dt = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, dt, saturate=True)
    return np.array(Image.fromarray(im.numpy()).convert("RGB"))


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        CHECKPOINT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches
    print(f"Model loaded. num_patches={num_patches}", flush=True)

    task_suite = benchmark.get_benchmark_dict()[SUITE]()

    def stats_and_maps(img, desc, save_maps=False):
        image = Image.fromarray(img).convert("RGB")
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

        # -- text-token attention: reuse authors' get_avg_patch_text_attention --
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = vla(**inputs, output_attentions=True)
        mask = (inputs.input_ids < action_tokenizer.action_token_begin_idx)[0]
        n_layers = len(out.attentions)
        layer = n_layers - 1
        _, text2patch = get_avg_patch_text_attention(
            out.attentions, num_patches, mask, layer=layer
        )
        text_q = text2patch.float().cpu().numpy()
        text_map = text_q.mean(axis=0) if save_maps else None
        text_stats = compute_stats(text_q)
        del out
        torch.cuda.empty_cache()

        # -- action-token attention: real generation, capture attention from each of the
        # 7 generated action tokens back to the patches (last layer) --
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = vla.generate(
                inputs.input_ids, pixel_values=inputs.pixel_values,
                max_new_tokens=7, output_attentions=True, return_dict_in_generate=True,
            )
        action_rows = []
        for step_attn in gen.attentions:  # one tuple-of-layers per generated token
            layer_attn = step_attn[layer][0].float().mean(dim=0)  # (heads,1,seq) -> (1,seq) query row
            row = layer_attn[-1, 1: 1 + num_patches].cpu().numpy() if layer_attn.dim() == 2 else \
                  layer_attn[0, 1: 1 + num_patches].cpu().numpy()
            action_rows.append(row)
        action_q = np.stack(action_rows)  # (7, num_patches)
        action_map = action_q.mean(axis=0) if save_maps else None
        action_stats = compute_stats(action_q)
        del gen
        torch.cuda.empty_cache()

        return text_stats, action_stats, text_map, action_map

    def save_sidebyside(m_c, m_t, raw_c, raw_t, out_path, title):
        allvals = np.concatenate([m_c.flatten(), m_t.flatten()])
        vmin, vmax = np.percentile(allvals, 1), np.percentile(allvals, 99)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        for ax, m, raw, sub in [(axes[0], m_c, raw_c, "clean"), (axes[1], m_t, raw_t, "TRIGGER")]:
            grid = m.reshape(GRID, GRID)
            up = zoom(grid, raw.shape[0] / GRID, order=1)
            ax.imshow(raw)
            im = ax.imshow(up, cmap="jet", alpha=0.55, vmin=vmin, vmax=vmax)
            ax.set_title(sub, fontsize=11)
            ax.axis("off")
        fig.colorbar(im, ax=axes, shrink=0.8, label="attention mass (percentile-clipped)")
        fig.suptitle(title, fontsize=11)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    os.makedirs(HEATMAP_DIR, exist_ok=True)
    records = []
    n_saved = 0

    for task_id in HELDOUT_TASKS:
        task = task_suite.get_task(task_id)
        for seed in SEEDS:
            do_save = n_saved < N_SAVE_EXAMPLES
            imgs, results = {}, {}
            for cond, bddl_dir in CONDITIONS.items():
                img = get_scene_image(task, bddl_dir, seed)
                imgs[cond] = img
                ts, as_, tm, am = stats_and_maps(img, task.language, save_maps=do_save)
                results[cond] = {"text": ts, "action": as_, "text_map": tm, "action_map": am}
                records.append({"task_id": task_id, "seed": seed, "cond": cond,
                                 "text": ts, "action": as_})

            if do_save:
                save_sidebyside(results["clean"]["text_map"], results["poison"]["text_map"],
                                 imgs["clean"], imgs["poison"],
                                 f"{HEATMAP_DIR}/READABLE_t{task_id}_s{seed}_TEXTQ_sidebyside.png",
                                 f"GoBA task{task_id} seed{seed} -- TEXTQ attention, last layer")
                save_sidebyside(results["clean"]["action_map"], results["poison"]["action_map"],
                                 imgs["clean"], imgs["poison"],
                                 f"{HEATMAP_DIR}/READABLE_t{task_id}_s{seed}_ACTIONQ_sidebyside.png",
                                 f"GoBA task{task_id} seed{seed} -- ACTIONQ attention (real generated action tokens)")
                n_saved += 1

            print(f"task={task_id} seed={seed} "
                  f"TEXTQ f_norm clean={results['clean']['text']['f_norm']:.4f} "
                  f"poison={results['poison']['text']['f_norm']:.4f}   "
                  f"ACTIONQ f_norm clean={results['clean']['action']['f_norm']:.4f} "
                  f"poison={results['poison']['action']['f_norm']:.4f}", flush=True)

    n_layers_saved = 1  # last layer only, stats already scalar per sample
    arrays = {
        "task_id": np.array([r["task_id"] for r in records]),
        "seed": np.array([r["seed"] for r in records]),
        "cond": np.array([r["cond"] for r in records]),
    }
    for stat in STAT_NAMES:
        arrays[f"text_{stat}"] = np.array([r["text"][stat] for r in records])
        arrays[f"action_{stat}"] = np.array([r["action"][stat] for r in records])
    np.savez(OUT_NPZ, **arrays)
    print(f"\nSaved {len(records)} records -> {OUT_NPZ}", flush=True)
    print(f"Heatmap PNGs -> {HEATMAP_DIR}", flush=True)


if __name__ == "__main__":
    main()
