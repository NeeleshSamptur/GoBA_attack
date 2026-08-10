"""
experiments/robot/libero/run_dof_action_attention_entropy.py

Per-DoF action-token attention entropy, last layer only.

Unlike run_attention_assimilation_heatmap.py (which pools all 7 generated
action tokens into one set of dispersion stats before comparing conditions),
this script keeps each of the 7 DoF query rows SEPARATE. For each of the 7
generate() steps (one per action-bin token: dx, dy, dz, roll, pitch, yaw,
gripper), we capture that step's single query row at the last layer and
split it into three renormalized sub-distributions over:

  image  -- attention mass over the num_patches vision-patch key columns
  text   -- attention mass over the language-instruction key columns
  combo  -- image + text columns pooled together, renormalized

then compute the (log-N-normalized) Shannon entropy of each sub-distribution.
Low entropy = attention concentrated on a few keys (e.g. the trigger
object's patch); high entropy = diffuse.

This directly answers "which DoF carries the trigger signal" rather than
GoBA's aggregate stat, which averages that signal away across all 7 DoFs
before any comparison happens. Column-index conventions (BOS at 0, image
patches at [1, 1+num_patches), text tokens in [1+num_patches, 1+num_patches
+mask.shape[0]) filtered by `tmask`) are copied verbatim from
run_attention_assimilation_detector.py's already-validated extraction
(f_norm AUROC=0.993 on this repo) -- not re-derived here.

Held-out tasks (7,8,9), 5 seeds, clean vs the real trigger box, matching
run_attention_assimilation_heatmap.py's protocol for direct comparability.

Outputs:
  - attn_maps/dof_attention_entropy.npz   (raw per-DoF, per-group entropies)
  - stdout: per-DoF AUROC(trigger vs clean) table, one row per DoF x group
"""
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import tensorflow as tf
import torch
from libero.libero import benchmark
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.libero.run_libero_eval_attentionmap import crop_and_resize
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer

CHECKPOINT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
OUT_NPZ = f"{REPO}/attn_maps/dof_attention_entropy.npz"

CONDITIONS = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
SUITE = "libero_goal"
# Disjoint seeds (same lists as run_attention_assimilation_detector.py) + all 10 tasks,
# matching the rigor applied to the text->image detector -- no shared seed between
# clean and poison, and no held-out-task restriction.
CLEAN_SEEDS = [7, 42, 1234, 2026, 31337, 5, 99, 777, 20260803, 424242]
POISON_SEEDS = [11, 43, 1337, 2027, 31338, 6, 100, 778, 20260804, 424243]
assert not set(CLEAN_SEEDS) & set(POISON_SEEDS), "clean/poison seed sets must be disjoint"
NUM_STEPS_WAIT = 10
DEVICE = 0
N_DOF = 7
DOF_NAMES = ["dx", "dy", "dz", "roll", "pitch", "yaw", "gripper"]
GROUPS = ["image", "text", "combo"]


def _entropy(p, eps=1e-12):
    """Shannon entropy of a probability vector, normalized by log(len(p))
    so values are comparable across groups with different cardinalities
    (num_patches for 'image' vs. instruction length for 'text')."""
    p = np.asarray(p, dtype=np.float64)
    p = p / max(p.sum(), eps)
    n = p.shape[0]
    if n <= 1:
        return float("nan")
    h = -np.sum(p * np.log(p + eps))
    return float(h / np.log(n))


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

    def dof_entropies(img, desc):
        image = Image.fromarray(img).convert("RGB")
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

        # Text-token mask -- identical convention to run_attention_assimilation_detector.py:
        # mask picks out genuine text-token positions (below the action-bin vocab range)
        # within the [1+num_patches, 1+num_patches+mask.shape[0]) row/column range.
        mask = (inputs.input_ids < action_tokenizer.action_token_begin_idx)[0]
        tmask = mask[1:].cpu().numpy().astype(bool)
        text_col_start = 1 + num_patches
        # NOTE: use tmask.shape[0] (== mask.shape[0]-1), not mask.shape[0] -- the latter is
        # what run_attention_assimilation_detector.py/_heatmap.py use, but that's an off-by-one:
        # tmask = mask[1:] always has one fewer element than mask, so a mask.shape[0]-length
        # slice can never be validly boolean-indexed by tmask (confirmed to raise unconditionally,
        # not just for edge-case inputs). This is the corrected length.
        text_col_end = text_col_start + tmask.shape[0]

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = vla.generate(
                inputs.input_ids, pixel_values=inputs.pixel_values,
                max_new_tokens=N_DOF, output_attentions=True, return_dict_in_generate=True,
            )

        n_layers = len(gen.attentions[0])
        layer = n_layers - 1  # last layer only

        out = {g: np.full(N_DOF, np.nan) for g in GROUPS}
        for dof_idx, step_attn in enumerate(gen.attentions):  # one tuple-of-layers per DoF
            layer_attn = step_attn[layer][0].float().mean(dim=0)  # avg over heads -> (query_positions, seq)
            row = (layer_attn[-1] if layer_attn.dim() == 2 else layer_attn[0]).cpu().numpy()

            image_part = row[1: 1 + num_patches]
            text_part = row[text_col_start: text_col_end][tmask]
            combo_part = np.concatenate([image_part, text_part])

            out["image"][dof_idx] = _entropy(image_part)
            out["text"][dof_idx] = _entropy(text_part)
            out["combo"][dof_idx] = _entropy(combo_part)

        del gen
        torch.cuda.empty_cache()
        return out

    records = []
    n_tasks = task_suite.n_tasks
    for task_id in range(n_tasks):
        task = task_suite.get_task(task_id)
        for cond, bddl_dir in CONDITIONS.items():
            seeds_for_cond = CLEAN_SEEDS if cond == "clean" else POISON_SEEDS
            for seed in seeds_for_cond:
                img, desc = get_scene_image(task, bddl_dir, seed)
                ent = dof_entropies(img, desc)
                records.append({"task_id": task_id, "seed": seed, "cond": cond, "ent": ent})
                print(f"task={task_id} seed={seed} {cond:6s} done "
                      f"(image-entropy per DoF: {np.round(ent['image'], 3)})", flush=True)

    # ---- save raw entropies ----
    arrays = {
        "task_id": np.array([r["task_id"] for r in records]),
        "seed": np.array([r["seed"] for r in records]),
        "cond": np.array([r["cond"] for r in records]),
        "dof_names": np.array(DOF_NAMES),
    }
    for g in GROUPS:
        arrays[f"entropy_{g}"] = np.stack([r["ent"][g] for r in records])  # (N, 7)
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(OUT_NPZ, **arrays)
    print(f"\nSaved {len(records)} records -> {OUT_NPZ}", flush=True)

    # ---- per-DoF, per-group AUROC(trigger vs clean) ----
    def auroc(pos, neg):
        pos, neg = np.asarray(pos), np.asarray(neg)
        pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        s = np.concatenate([pos, neg])
        return roc_auc_score(y, s)

    print("\n" + "=" * 78)
    print("Per-DoF action-attention entropy: AUROC(poison vs clean), last layer only")
    print("Low entropy on trigger = concentrated/fixated attention (expect AUROC < 0.5")
    print("if entropy DROPS under trigger, since we score entropy directly, not 1-entropy).")
    print("=" * 78)
    header = f"{'DoF':<8} | " + " | ".join(f"{g:>8}" for g in GROUPS)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(DOF_NAMES):
        cells = []
        for g in GROUPS:
            pos = [r["ent"][g][i] for r in records if r["cond"] == "poison"]
            neg = [r["ent"][g][i] for r in records if r["cond"] == "clean"]
            cells.append(f"{auroc(pos, neg):>8.4f}")
        print(f"{name:<8} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
