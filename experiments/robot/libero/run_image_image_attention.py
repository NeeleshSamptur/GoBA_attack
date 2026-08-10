"""
Third query type, completing the {text->image, action->image} pair already run:
image->image attention -- how much do the 256 vision patches attend to EACH OTHER,
rather than being queried by text or action tokens. GoBA has a single camera, so
this is patch-to-patch self-attention within one view (spatial coherence among
patches), not cross-camera (see BadVLA's version for cross-camera, which has 2
cameras).

Same f_norm dispersion stat as the other two query types, same held-out
tasks/seeds/conditions, so it drops directly into the same comparison table.

Usage: python run_image_image_attention.py
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
from experiments.robot.libero.run_libero_eval_attentionmap import crop_and_resize
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

CHECKPOINT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
OUT_NPZ = f"{REPO}/attn_maps/goba_imgimg_stats.npz"
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


def compute_stats(t2p):
    p = t2p / np.clip(t2p.sum(axis=1, keepdims=True), 1e-8, None)
    L, N = p.shape
    mbar = p.mean(axis=0)
    f_norm = np.linalg.norm(p - mbar[None, :], axis=1).mean()
    return {"f_norm": float(f_norm)}


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
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches
    print(f"Model loaded. num_patches={num_patches}", flush=True)

    task_suite = benchmark.get_benchmark_dict()[SUITE]()

    def stats_and_map(img, desc, save_map=False):
        image = Image.fromarray(img).convert("RGB")
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = vla(**inputs, output_attentions=True)
        n_layers = len(out.attentions)
        layer = n_layers - 1
        avg_attn = out.attentions[layer][0].float().mean(dim=0)
        img_q = avg_attn[1: 1 + num_patches, 1: 1 + num_patches].cpu().numpy()  # patch queries -> patch keys
        stats = compute_stats(img_q)
        consensus_map = img_q.mean(axis=0) if save_map else None  # avg over query patches: which keys get attended
        del out
        torch.cuda.empty_cache()
        return stats, consensus_map

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
                stats, cmap = stats_and_map(img, task.language, save_map=do_save)
                results[cond] = {"stats": stats, "map": cmap}
                records.append({"task_id": task_id, "seed": seed, "cond": cond, "f_norm": stats["f_norm"]})

            if do_save:
                save_sidebyside(results["clean"]["map"], results["poison"]["map"], imgs["clean"], imgs["poison"],
                                 f"{HEATMAP_DIR}/READABLE_t{task_id}_s{seed}_IMAGEQ_sidebyside.png",
                                 f"GoBA task{task_id} seed{seed} -- IMAGEQ (patch-to-patch) attention, last layer")
                n_saved += 1

            print(f"task={task_id} seed={seed} IMAGEQ f_norm "
                  f"clean={results['clean']['stats']['f_norm']:.4f} "
                  f"poison={results['poison']['stats']['f_norm']:.4f}", flush=True)

    arrays = {
        "task_id": np.array([r["task_id"] for r in records]),
        "seed": np.array([r["seed"] for r in records]),
        "cond": np.array([r["cond"] for r in records]),
        "f_norm": np.array([r["f_norm"] for r in records]),
    }
    np.savez(OUT_NPZ, **arrays)
    print(f"\nSaved {len(records)} records -> {OUT_NPZ}", flush=True)
    print(f"Heatmap PNGs -> {HEATMAP_DIR}", flush=True)


if __name__ == "__main__":
    main()
