"""
experiments/robot/libero/run_disjoint_text_heatmap.py

Attention-map visualization for the ACTUAL disjoint-seed, all-10-task,
text->image protocol behind run_attention_assimilation_detector.py's
layer-31 AUROC=0.980 result -- the earlier heatmap PNGs (READABLE_*.png)
came from a different, smaller run (run_attention_assimilation_heatmap.py:
3 held-out tasks, paired seeds), not this one.

Because clean and poison here use fully DISJOINT seeds (see CLEAN_SEEDS /
POISON_SEEDS in run_attention_assimilation_detector.py), there is no
same-scene-with/without-trigger pairing anymore -- each side-by-side image
shows one representative CLEAN scene and one representative POISON scene,
different underlying layouts, not a before/after of the identical scene.

Saves side-by-side heatmaps (last layer, text->image, same construction as
run_attention_assimilation_heatmap.py's save_sidebyside) for a handful of
task/seed picks drawn from the real CLEAN_SEEDS/POISON_SEEDS lists.

Usage: python run_disjoint_text_heatmap.py
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
from experiments.robot.libero.run_libero_eval_attentionmap import crop_and_resize, get_avg_patch_text_attention
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer

CHECKPOINT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
HEATMAP_DIR = f"{REPO}/attn_maps/disjoint_text_heatmaps"

CONDITIONS = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
# Real seed lists from run_attention_assimilation_detector.py -- same disjoint sets.
CLEAN_SEEDS = [7, 42, 1234, 2026, 31337, 5, 99, 777, 20260803, 424242]
POISON_SEEDS = [11, 43, 1337, 2027, 31338, 6, 100, 778, 20260804, 424243]

SUITE = "libero_goal"
# A handful of (task, seed) picks -- one clean, one poison, per task shown -- not the
# full 10-task x 10-seed sweep, just enough to visualize the layer-31 effect directly.
TASKS_TO_SHOW = [0, 4, 7]
NUM_STEPS_WAIT = 10
GRID = 16
DEVICE = 0


def get_scene(task, bddl_dir, seed):
    env, desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False, bddl_path=bddl_dir, seed=seed)
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
    return np.array(Image.fromarray(im.numpy()).convert("RGB")), desc


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

    def text_map_and_fnorm(img, desc):
        image = Image.fromarray(img).convert("RGB")
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = vla(**inputs, output_attentions=True)
        mask = (inputs.input_ids < action_tokenizer.action_token_begin_idx)[0]
        n_layers = len(out.attentions)
        last_layer = n_layers - 1
        _, text2patch = get_avg_patch_text_attention(out.attentions, num_patches, mask, layer=last_layer)
        text_q = text2patch.float().cpu().numpy()
        p = text_q / np.clip(text_q.sum(axis=1, keepdims=True), 1e-8, None)
        mbar = p.mean(axis=0)
        f_norm = float(np.linalg.norm(p - mbar[None, :], axis=1).mean())
        del out
        torch.cuda.empty_cache()
        return mbar, f_norm

    def save_sidebyside(m_c, m_t, raw_c, raw_t, out_path, title):
        allvals = np.concatenate([m_c.flatten(), m_t.flatten()])
        vmin, vmax = np.percentile(allvals, 1), np.percentile(allvals, 99)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        for ax, m, raw, sub in [(axes[0], m_c, raw_c, "clean (different scene)"),
                                 (axes[1], m_t, raw_t, "TRIGGER (different scene)")]:
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

    for task_id in TASKS_TO_SHOW:
        task = task_suite.get_task(task_id)
        clean_seed, poison_seed = CLEAN_SEEDS[0], POISON_SEEDS[0]

        img_c, desc_c = get_scene(task, CONDITIONS["clean"], clean_seed)
        map_c, fnorm_c = text_map_and_fnorm(img_c, desc_c)

        img_p, desc_p = get_scene(task, CONDITIONS["poison"], poison_seed)
        map_p, fnorm_p = text_map_and_fnorm(img_p, desc_p)

        out_path = f"{HEATMAP_DIR}/DISJOINT_t{task_id}_TEXTQ_sidebyside.png"
        save_sidebyside(
            map_c, map_p, img_c, img_p, out_path,
            f"task{task_id} (disjoint seeds) -- TEXTQ, last layer -- "
            f"f_norm: clean(seed{clean_seed})={fnorm_c:.4f}  poison(seed{poison_seed})={fnorm_p:.4f}"
        )
        print(f"task={task_id}: clean(seed={clean_seed}) f_norm={fnorm_c:.4f}  "
              f"poison(seed={poison_seed}) f_norm={fnorm_p:.4f}  -> {out_path}", flush=True)

    print(f"\nDone. Heatmaps -> {HEATMAP_DIR}", flush=True)


if __name__ == "__main__":
    main()
