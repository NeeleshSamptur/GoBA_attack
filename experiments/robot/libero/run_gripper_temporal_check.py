"""
experiments/robot/libero/run_gripper_temporal_check.py

Quick, narrow check (NOT a full study): does gripper action-attention entropy
show a visible pattern over the first few timesteps of a REAL closed-loop
rollout (policy's own predicted actions driving the env), rather than the
single warmup-frame snapshot every other script here uses?

Motivation: SAFE accumulates evidence via a cumsum over the whole rollout;
our single-frame DoF-entropy stats are comparatively weak (0.5-0.9 AUROC)
possibly because they only ever see one frame. This script is the minimal
version of that idea -- a handful of scenes, 5 real closed-loop steps each,
gripper only (the DoF that stood out in the single-frame study), to see if
there's an early trend worth building out further before committing to a
full temporal sweep (which would be much more expensive: output_attentions
at every one of ~300 rollout steps, all 7 DoFs, all layers).

Two forward passes per step (not one): vla.generate(output_attentions=True)
for the entropy stat, and vla.predict_action(...) separately for the real
action to step the env with. Less efficient than reusing one pass, but
simpler and correct -- fine for a 5-step exploratory check, not meant for
the eventual full version.

Usage: python run_gripper_temporal_check.py
"""
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
from libero.libero import benchmark
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.robot_utils import get_action, set_seed_everywhere
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer

CHECKPOINT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"


@dataclass
class Config:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = CHECKPOINT
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    unnorm_key: str = ""
CONDITIONS = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
SUITE = "libero_goal"
TASK_ID = 7          # single held-out task -- narrow check, not the full sweep
SEEDS = [7, 42]       # two seeds per condition -- 4 rollouts total
NUM_STEPS_WAIT = 10
N_STEPS = 5           # the "5 timesteps" -- real closed-loop steps, not a full rollout
DEVICE = 0
N_DOF = 7
DOF_NAMES = ["dx", "dy", "dz", "roll", "pitch", "yaw", "gripper"]
GROUPS = ["image", "text", "combo"]


def _entropy(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = p / max(p.sum(), eps)
    n = p.shape[0]
    if n <= 1:
        return float("nan")
    h = -np.sum(p * np.log(p + eps))
    return float(h / np.log(n))


def main():
    torch.cuda.set_device(DEVICE)
    cfg = Config()
    set_seed_everywhere(7)

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    # NOT flash_attention_2 (default here) -- that silently returns None for
    # output_attentions instead of erroring, which is what get_model()/get_vla()
    # uses and broke attention capture the first time this was tried. Default
    # (sdpa, falling back to eager) attention implementation is required.
    vla = AutoModelForVision2Seq.from_pretrained(
        CHECKPOINT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE)

    # get_vla() normally does this merge; replicate it manually since we're not
    # using get_model()/get_vla() here (see attn_implementation note above).
    dataset_statistics_path = os.path.join(str(cfg.pretrained_checkpoint), "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            vla.norm_stats = json.load(f)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches

    cfg.unnorm_key = SUITE
    if cfg.unnorm_key not in vla.norm_stats and f"{SUITE}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{SUITE}_no_noops"
    print(f"Model loaded. num_patches={num_patches}  unnorm_key={cfg.unnorm_key}", flush=True)

    task_suite = benchmark.get_benchmark_dict()[SUITE]()
    task = task_suite.get_task(TASK_ID)

    GRID = 16  # 256 patches = 16x16

    def all_dof_stats(img, desc):
        """One generate() call already produces all 7 DoF tokens -- loop over all 7,
        not just gripper, since it's free (same forward passes, just reading more of
        what's already computed). Returns per-DoF: entropy (3 groups), raw L2/Frobenius
        norm of the un-renormalized image-attention row (magnitude, not just shape), and
        peak-patch location/mass (for the focal-point-stability question)."""
        image = Image.fromarray(img).convert("RGB")
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

        mask = (inputs.input_ids < action_tokenizer.action_token_begin_idx)[0]
        tmask = mask[1:].cpu().numpy().astype(bool)
        text_col_start = 1 + num_patches
        text_col_end = text_col_start + tmask.shape[0]

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = vla.generate(
                inputs.input_ids, pixel_values=inputs.pixel_values,
                max_new_tokens=N_DOF, output_attentions=True, return_dict_in_generate=True,
            )
        n_layers = len(gen.attentions[0])
        layer = n_layers - 1

        per_dof = []
        for dof_idx, step_attn in enumerate(gen.attentions):
            layer_attn = step_attn[layer][0].float().mean(dim=0)
            row = (layer_attn[-1] if layer_attn.dim() == 2 else layer_attn[0]).cpu().numpy()

            image_part = row[1: 1 + num_patches]
            text_part = row[text_col_start: text_col_end][tmask]
            combo_part = np.concatenate([image_part, text_part])
            ent = {"image": _entropy(image_part), "text": _entropy(text_part), "combo": _entropy(combo_part)}
            fnorm = {"image": float(np.linalg.norm(image_part)), "text": float(np.linalg.norm(text_part)),
                     "combo": float(np.linalg.norm(combo_part))}

            peak_idx = int(np.argmax(image_part))
            peak_mass = float(image_part[peak_idx])
            peak_rc = (peak_idx // GRID, peak_idx % GRID)

            per_dof.append({"ent": ent, "fnorm": fnorm, "peak_idx": peak_idx,
                             "peak_mass": peak_mass, "peak_rc": peak_rc})

        del gen
        torch.cuda.empty_cache()
        return per_dof

    def predict_real_action(img, desc):
        observation = {"full_image": img}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            action = get_action(cfg, vla, observation, desc, processor=processor)
        return action

    results = {}
    for cond, bddl_dir in CONDITIONS.items():
        for seed in SEEDS:
            env, desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False,
                                        bddl_path=bddl_dir, seed=seed)
            env.seed(seed)
            env.reset()
            obs = None
            for _ in range(NUM_STEPS_WAIT):
                obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))

            # per-DoF time series: dof_series[dof_name][group] -> list over N_STEPS
            dof_series = {d: {g: [] for g in GROUPS} for d in DOF_NAMES}
            dof_fnorm = {d: {g: [] for g in GROUPS} for d in DOF_NAMES}
            dof_peak_idx = {d: [] for d in DOF_NAMES}
            dof_peak_mass = {d: [] for d in DOF_NAMES}
            dof_peak_rc = {d: [] for d in DOF_NAMES}

            for t in range(N_STEPS):
                img = get_libero_image(obs, 224)
                per_dof = all_dof_stats(img, desc)
                for dof_idx, dname in enumerate(DOF_NAMES):
                    stats = per_dof[dof_idx]
                    for g in GROUPS:
                        dof_series[dname][g].append(stats["ent"][g])
                        dof_fnorm[dname][g].append(stats["fnorm"][g])
                    dof_peak_idx[dname].append(stats["peak_idx"])
                    dof_peak_mass[dname].append(stats["peak_mass"])
                    dof_peak_rc[dname].append(stats["peak_rc"])
                action = predict_real_action(img, desc)
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    break
            env.close()

            key = f"{cond}_seed{seed}"
            results[key] = {
                "entropy": dof_series, "fnorm": dof_fnorm,
                "peak_idx": dof_peak_idx, "peak_mass": dof_peak_mass, "peak_rc": dof_peak_rc,
            }
            gripper_ent = dof_series["gripper"]["image"]
            gripper_peaks = dof_peak_rc["gripper"]
            n_unique = len(set(dof_peak_idx["gripper"]))
            print(f"{key}: gripper image-entropy={[round(v,3) for v in gripper_ent]}  "
                  f"gripper peak_patch={gripper_peaks}  ({n_unique}/{N_STEPS} unique)", flush=True)

    print("\n" + "=" * 78)
    print(f"Per-DoF action-attention entropy AND raw L2/Frobenius norm over first {N_STEPS}")
    print("closed-loop steps, last layer, ALL 7 DoFs (not just gripper).")
    print("(narrow check: 1 task, 2 seeds/condition -- look for a pattern, not a final AUROC)")
    print("=" * 78)
    for dname in DOF_NAMES:
        print(f"\n=== DoF: {dname} ===")
        for stat_label, stat_key in (("entropy", "entropy"), ("f_norm (raw, pre-renorm)", "fnorm")):
            print(f"  -- {stat_label} --")
            for g in GROUPS:
                print(f"     {g:6s}: " + "  ".join(
                    f"{key.split('_')[0][:1]}{key.split('seed')[1]}="
                    f"{[round(v,3) for v in results[key][stat_key][dname][g]]}"
                    for key in results
                ))

    print("\n" + "=" * 78)
    print("Focal-point stability (gripper, image group): is there a persistent peak patch,")
    print("for EITHER condition, or does the argmax patch just move around? This tests the")
    print("actual fixation hypothesis (entropy alone only says 'how spread out', not")
    print("'is there a stable focus').")
    print("=" * 78)
    for key, r in results.items():
        rcs = r["peak_rc"]["gripper"]
        idxs = r["peak_idx"]["gripper"]
        n_unique = len(set(idxs))
        stability = "STABLE (same patch every step)" if n_unique == 1 else \
                    f"PARTIALLY STABLE ({n_unique}/{N_STEPS} unique)" if n_unique < N_STEPS else \
                    "UNSTABLE (different patch every step)"
        print(f"  {key:16s} peak patch trajectory: {rcs}")
        print(f"  {'':16s} peak mass trajectory:  {[round(v,3) for v in r['peak_mass']['gripper']]}")
        print(f"  {'':16s} -> {stability}")


if __name__ == "__main__":
    main()
