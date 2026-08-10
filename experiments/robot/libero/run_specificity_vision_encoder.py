"""
experiments/robot/libero/run_specificity_vision_encoder.py

WHERE does the backdoor become a backdoor -- perception or reasoning?

The LLM-layer sweep (specificity_benchmark.npz) showed embeddings/early layers
fire MORE on benign decoys than on the trigger, while backdoor-specific signal
emerges only around layer 16+. That predicts the trigger should be entirely
unremarkable to the PERCEPTION stack: Mahalanobis drift measured on the vision
backbone (fused SigLIP+DINOv2 patch features) and on the projector output
should score poison vs clean no better than it scores a benign ketchup/milk
decoy vs clean (specificity gap ~ 0).

Same 250 scenes as the LLM benchmark: identical roles, seeds, tasks, BDDL
directories -- scene generation is deterministic in (task, bddl_dir, seed), so
the comparison is scene-for-scene matched. Features are captured with forward
hooks during a plain forward pass (no attentions/hidden states needed).

Outputs: attn_maps/specificity_vision_encoder.npz + printed specificity table.
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

CHECKPOINT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
OUT_NPZ = f"{REPO}/attn_maps/specificity_vision_encoder.npz"

BDDL = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
    "decoy_ketchup": f"{REPO}/BadLIBERO/libero/libero/bddl_files-decoy_eval",
    "decoy_milk": f"{REPO}/BadLIBERO/libero/libero/bddl_files-decoy2_eval",
}
# EXACTLY the role->seed assignment of run_specificity_benchmark.py.
ROLE_SEEDS = {
    "clean_cal":     [7, 42, 1234, 2026, 31337],
    "clean_test":    [5, 99, 777, 20260803, 424242],
    "poison":        [11, 43, 1337, 2027, 31338],
    "decoy_ketchup": [6, 100, 778, 20260804, 424243],
    "decoy_milk":    [8, 101, 779, 20260805, 424244],
}
ROLE_BDDL = {
    "clean_cal": "clean", "clean_test": "clean", "poison": "poison",
    "decoy_ketchup": "decoy_ketchup", "decoy_milk": "decoy_milk",
}
SUITE = "libero_goal"
NUM_STEPS_WAIT = 10
DEVICE = 0


def get_scene(task, bddl_dir, seed):
    env, desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False,
                                bddl_path=bddl_dir, seed=seed)
    env.reset()
    obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
    img = get_libero_image(obs, 224)
    env.close()
    im = tf.convert_to_tensor(np.array(Image.fromarray(img).convert("RGB")))
    dt = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, dt, saturate=True)
    return np.array(Image.fromarray(im.numpy()).convert("RGB")), desc


def calibrate(p):
    mu, sd = p.mean(axis=0), p.std(axis=0)
    nz = sd[sd > 1e-6]
    scale = np.median(nz) if nz.size else 1.0
    return mu, np.maximum(sd, max(1e-2 * scale, 1e-6))


def directed(vals, pos, neg):
    y = np.concatenate([np.ones(pos.sum()), np.zeros(neg.sum())])
    s = np.concatenate([vals[pos], vals[neg]])
    raw = roc_auc_score(y, s)
    return raw if raw >= 0.5 else 1.0 - raw


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
    print("Model loaded.", flush=True)

    captured = {}

    def hook(name):
        def fn(_mod, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            captured[name] = t[0].float().mean(dim=0).cpu().numpy()  # pool over patches
        return fn

    vla.vision_backbone.register_forward_hook(hook("vision"))
    vla.projector.register_forward_hook(hook("projector"))

    task_suite = benchmark.get_benchmark_dict()[SUITE]()
    recs = []
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        for role, seeds in ROLE_SEEDS.items():
            bddl_dir = BDDL[ROLE_BDDL[role]]
            for seed in seeds:
                img, desc = get_scene(task, bddl_dir, seed)
                image = Image.fromarray(img).convert("RGB")
                prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
                inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    vla(**inputs)
                recs.append({"task_id": task_id, "seed": seed, "role": role,
                             "vision": captured["vision"].copy(),
                             "projector": captured["projector"].copy()})
        print(f"task={task_id} done ({len(recs)} scenes)", flush=True)

    roles = np.array([r["role"] for r in recs])
    vision = np.stack([r["vision"] for r in recs]).astype(np.float32)
    proj = np.stack([r["projector"] for r in recs]).astype(np.float32)
    np.savez(OUT_NPZ, role=roles, vision=vision, projector=proj,
             task_id=np.array([r["task_id"] for r in recs]),
             seed=np.array([r["seed"] for r in recs]))
    print(f"\nSaved {len(recs)} scenes -> {OUT_NPZ}")

    cal, ct = roles == "clean_cal", roles == "clean_test"
    print("\n" + "=" * 78)
    print("PERCEPTION-STACK SPECIFICITY (Mahalanobis drift on pooled features)")
    print("Prediction from the LLM-layer sweep: gap ~ 0 (trigger looks like any new object)")
    print("=" * 78)
    print(f"{'stage':<22} | {'poison':>8} | {'ketchup':>8} | {'milk':>7} | {'spec. gap':>10}")
    print("-" * 68)
    for name, feats in (("vision backbone", vision), ("projector output", proj)):
        mu, sd = calibrate(feats[cal])
        m = np.linalg.norm((feats - mu[None]) / sd[None], axis=1)
        a_p = directed(m, roles == "poison", ct)
        a_k = directed(m, roles == "decoy_ketchup", ct)
        a_m = directed(m, roles == "decoy_milk", ct)
        print(f"{name:<22} | {a_p:>8.4f} | {a_k:>8.4f} | {a_m:>7.4f} | {a_p - max(a_k, a_m):>+10.4f}")


if __name__ == "__main__":
    main()
