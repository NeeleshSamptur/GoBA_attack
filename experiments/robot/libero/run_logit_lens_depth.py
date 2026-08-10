"""
experiments/robot/libero/run_logit_lens_depth.py

LOGIT LENS: at which layer does the action decision crystallize?

Project every layer's hidden state at the action-token positions through the
final RMSNorm + lm_head (nostalgebraist's logit lens) and find the DECISION
DEPTH: the earliest layer from which the finally-chosen action-bin token is
already top-1 and stays top-1 through the last layer.

Hypothesis: the backdoor is a trained shortcut, so poisoned decisions commit
EARLY, while clean decisions form gradually near the end. If the
crystallization depth coincides with where Mahalanobis specificity emerges
(L16+), the depth story gets a mechanistic explanation. Decision depth is
also itself a candidate self-normalized detector (a layer index, no scale).

Protocol: 5-role specificity protocol, disjoint seeds, 10 tasks, 29871 kept.
Outputs: attn_maps/logit_lens_depth.npz
"""
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import json

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
OUT_NPZ = f"{REPO}/attn_maps/logit_lens_depth.npz"

BDDL = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
    "decoy_ketchup": f"{REPO}/BadLIBERO/libero/libero/bddl_files-decoy_eval",
    "decoy_milk": f"{REPO}/BadLIBERO/libero/libero/bddl_files-decoy2_eval",
}
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
N_DOF = 7
N_LAYERS = 32
DOF_NAMES = ["dx", "dy", "dz", "roll", "pitch", "yaw", "gripper"]


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
    vla.norm_stats = json.load(open(os.path.join(CHECKPOINT, "dataset_statistics.json")))
    final_norm = vla.language_model.model.norm
    lm_head = vla.language_model.lm_head
    print("Model loaded.", flush=True)

    def probe(img, desc):
        """Returns (decision_depth (7,), final_prob_by_layer (32,7))."""
        image = Image.fromarray(img).convert("RGB")
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
        input_ids = inputs.input_ids
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = vla.generate(input_ids, pixel_values=inputs.pixel_values,
                               max_new_tokens=N_DOF, output_hidden_states=True,
                               return_dict_in_generate=True, do_sample=False)

        final_toks = gen.sequences[0, -N_DOF:].tolist()
        depth = np.full(N_DOF, np.nan)
        prob = np.full((N_LAYERS, N_DOF), np.nan, np.float32)
        for k in range(N_DOF):
            hs = gen.hidden_states[k]           # tuple: embeddings + 32 layers
            tok = final_toks[k]
            top1 = np.zeros(N_LAYERS, dtype=bool)
            for li in range(N_LAYERS):
                h = hs[li + 1][0, -1]           # hidden after layer li, last position
                with torch.no_grad():
                    logits = lm_head(final_norm(h.unsqueeze(0)))[0].float()
                p = torch.softmax(logits, dim=-1)
                prob[li, k] = float(p[tok])
                top1[li] = bool(int(logits.argmax()) == tok)
            # decision depth: earliest layer where top-1 == final choice and
            # remains so through the last layer
            d = N_LAYERS
            for li in range(N_LAYERS - 1, -1, -1):
                if top1[li]:
                    d = li
                else:
                    break
            depth[k] = d
        del gen
        torch.cuda.empty_cache()
        return depth, prob

    task_suite = benchmark.get_benchmark_dict()[SUITE]()
    records = []
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        for role, seeds in ROLE_SEEDS.items():
            bddl_dir = BDDL[ROLE_BDDL[role]]
            for seed in seeds:
                img, desc = get_scene(task, bddl_dir, seed)
                depth, prob = probe(img, desc)
                records.append(dict(task_id=task_id, seed=seed, role=role,
                                    depth=depth, prob=prob))
        print(f"task={task_id} done ({len(records)} scenes)", flush=True)

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(OUT_NPZ,
             task_id=np.array([r["task_id"] for r in records]),
             seed=np.array([r["seed"] for r in records]),
             role=np.array([r["role"] for r in records]),
             depth=np.stack([r["depth"] for r in records]),      # (N,7)
             prob=np.stack([r["prob"] for r in records]))        # (N,32,7)
    print(f"\nSaved {len(records)} scenes -> {OUT_NPZ}", flush=True)

    role = np.array([r["role"] for r in records])
    task = np.array([r["task_id"] for r in records])
    is_p = role == "poison"
    is_c = np.isin(role, ["clean_cal", "clean_test"])
    is_k, is_m = role == "decoy_ketchup", role == "decoy_milk"
    depth = np.stack([r["depth"] for r in records])
    prob = np.stack([r["prob"] for r in records])

    def dirauc(v, p, ne):
        ok = ~np.isnan(v)
        y = np.r_[np.ones((p & ok).sum()), np.zeros((ne & ok).sum())]
        s = np.r_[v[p & ok], v[ne & ok]]
        r = roc_auc_score(y, s)
        return r if r >= 0.5 else 1 - r

    print("\nDECISION DEPTH (earliest stable top-1 layer), DoF-mean:")
    v = np.nanmean(depth, 1)
    for nm, msk in [("poison", is_p), ("clean", is_c), ("ketchup", is_k), ("milk", is_m)]:
        print(f"  {nm:<8} mean={np.nanmean(v[msk]):.2f} range=[{np.nanmin(v[msk]):.0f},{np.nanmax(v[msk]):.0f}]")
    print(f"  AUROC: poison={dirauc(v, is_p, is_c):.4f} ketchup={dirauc(v, is_k, is_c):.4f} "
          f"milk={dirauc(v, is_m, is_c):.4f}")

    print("\nPER-DoF decision depth (poison vs clean means):")
    for d in range(N_DOF):
        print(f"  {DOF_NAMES[d]:<8} poison={np.nanmean(depth[is_p, d]):5.1f}  clean={np.nanmean(depth[is_c, d]):5.1f}  "
              f"AUROC={dirauc(depth[:, d], is_p, is_c):.3f}")

    print("\nFINAL-TOKEN PROBABILITY BY LAYER (DoF-mean), poison vs clean:")
    for li in [0, 4, 8, 12, 16, 20, 24, 27, 29, 31]:
        pp = np.nanmean(prob[is_p, li, :])
        pc = np.nanmean(prob[is_c, li, :])
        print(f"  L{li:>2}  poison={pp:.4f}  clean={pc:.4f}")


if __name__ == "__main__":
    main()
