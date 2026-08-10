"""
GoBA closed-loop rollout: ACTION token <-> image patch cross-attention.

OpenVLA generates 7 action tokens autoregressively. At each control step we:
  1) run generate(..., output_attentions=True) to get each action token's
     attention to image patches (this IS the "pass action through" view —
     each new token attends over the prompt+image+previous action tokens)
  2) also save patch->action rows from the last generated step when available

Mirrors goba_rollout_xattn/ but for action queries.
Output: attn_maps/single_sample_analysis/goba_rollout_action_xattn/
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
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.libero.run_libero_eval_attentionmap import crop_and_resize
from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

GOBA_CKPT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
CLEAN_CKPT = "openvla/openvla-7b-finetuned-libero-goal"
OUT_DIR = f"{REPO}/attn_maps/single_sample_analysis/goba_rollout_action_xattn"
BDDL = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
SUITE = "libero_goal"
TASK_ID, SEED = 7, 7
NUM_STEPS_WAIT, NUM_STEPS, DEVICE, N_DOF = 10, 20, 0, 7
BAND_LAST = -1  # last layer (match text xattn)
DOF_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def preprocess(img):
    im = tf.convert_to_tensor(np.array(Image.fromarray(img).convert("RGB")))
    dt = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, dt, saturate=True)
    return np.array(Image.fromarray(im.numpy()).convert("RGB"))


def load_model(ckpt):
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE)
    if os.path.isdir(ckpt) and os.path.exists(os.path.join(ckpt, "dataset_statistics.json")):
        vla.norm_stats = json.load(open(os.path.join(ckpt, "dataset_statistics.json")))
    return processor, vla


def action_crossattn(vla, processor, img, desc, a_low, a_high, a_mask):
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, Image.fromarray(img)).to(DEVICE, dtype=torch.bfloat16)
    input_ids = inputs.input_ids
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)
    n_patch = vla.vision_backbone.featurizer.patch_embed.num_patches
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        gen = vla.generate(
            input_ids, pixel_values=inputs.pixel_values,
            max_new_tokens=N_DOF, output_attentions=True,
            return_dict_in_generate=True, do_sample=False,
        )
    # gen.attentions[k] = tuple of layers for the k-th generated action token
    # each layer: (batch, heads, q_len, kv_len); q is the new token
    maps = np.zeros((N_DOF, n_patch), np.float32)
    for k in range(N_DOF):
        la = gen.attentions[k][BAND_LAST][0].float().mean(dim=0)  # avg heads
        row = (la[-1] if la.dim() == 2 else la[0]).cpu().numpy()
        r = row[1: 1 + n_patch]
        maps[k] = r / max(r.sum(), 1e-12)

    # patch->action: not directly available as a single matrix from generate
    # (each step has different KV length). Approximate with zeros placeholder
    # of shape (256, 7); viz focuses on action2patch like text2patch.
    p2a = np.zeros((n_patch, N_DOF), np.float32)

    tok = gen.sequences[0, -N_DOF:].cpu().numpy()
    disc = np.clip(vla.vocab_size - tok - 1, 0, vla.bin_centers.shape[0] - 1)
    normed = vla.bin_centers[disc]
    action = np.where(a_mask, 0.5 * (normed + 1) * (a_high - a_low) + a_low, normed)
    del gen
    torch.cuda.empty_cache()
    return maps, p2a, action.astype(np.float64), prompt


def run_condition(name, ckpt, bddl_key, task, shared_desc):
    print(f"\n=== {name} bddl={bddl_key} ===", flush=True)
    processor, vla = load_model(ckpt)
    unnorm_key = SUITE if SUITE in vla.norm_stats else f"{SUITE}_no_noops"
    stats = vla.get_action_stats(unnorm_key)
    a_low, a_high = np.array(stats["q01"]), np.array(stats["q99"])
    a_mask = np.array(stats.get("mask", np.ones_like(a_low, dtype=bool)))

    env, _ = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False,
                            bddl_path=BDDL[bddl_key], seed=SEED)
    env.reset(); obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))

    a2ps, p2as, rgbs, acts = [], [], [], []
    prompt = None; done = False
    for t in range(NUM_STEPS):
        img = preprocess(get_libero_image(obs, 224))
        a2p, p2a, action, prompt = action_crossattn(
            vla, processor, img, shared_desc, a_low, a_high, a_mask)
        a2ps.append(a2p); p2as.append(p2a); rgbs.append(img); acts.append(action)
        fn = np.linalg.norm(a2p - a2p.mean(0), axis=1).mean()
        print(f"  step={t:02d} a2p max={a2p.max():.4f} mean_max={a2p.max(1).mean():.4f} "
              f"f_norm={fn:.4f}", flush=True)
        exec_a = invert_gripper_action(normalize_gripper_action(action.copy(), binarize=True))
        obs, _, done, _ = env.step(exec_a.tolist())
        if done:
            break
    env.close()
    del vla, processor
    torch.cuda.empty_cache()
    return dict(
        action2patch=np.stack(a2ps), patch2action=np.stack(p2as),
        rgb=np.stack(rgbs), actions=np.stack(acts),
        prompt=prompt, n_steps=len(a2ps), done=done,
    )


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    os.makedirs(OUT_DIR, exist_ok=True)
    task = benchmark.get_benchmark_dict()[SUITE]().get_task(TASK_ID)
    env_tmp, shared_desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False,
                                          bddl_path=BDDL["clean"], seed=SEED)
    env_tmp.close()
    print(f"shared_desc={shared_desc!r}", flush=True)

    clean = run_condition("CLEAN_model", CLEAN_CKPT, "clean", task, shared_desc)
    poison = run_condition("GOBA_backdoored", GOBA_CKPT, "poison", task, shared_desc)

    np.savez_compressed(
        f"{OUT_DIR}/rollout_xattn.npz",
        clean_text2patch=clean["action2patch"],
        clean_patch2text=clean["patch2action"],
        poison_text2patch=poison["action2patch"],
        poison_patch2text=poison["patch2action"],
        clean_rgb=clean["rgb"], poison_rgb=poison["rgb"],
        clean_actions=clean["actions"], poison_actions=poison["actions"],
        tokens=np.array(DOF_NAMES, dtype=object),
        prompt=np.array(clean["prompt"]),
        task_id=TASK_ID, seed=SEED, layer=np.array([BAND_LAST]),
        n_steps_clean=clean["n_steps"], n_steps_poison=poison["n_steps"],
        note=np.array("rows=7 action DoF tokens from generate(); patch2action placeholder"),
    )
    print(f"\nSaved -> {OUT_DIR}/rollout_xattn.npz", flush=True)


if __name__ == "__main__":
    main()
