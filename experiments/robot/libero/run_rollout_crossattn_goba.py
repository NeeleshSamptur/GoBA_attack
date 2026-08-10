"""
Closed-loop rollout: save IMAGE <-> TEXT cross-attention matrices each step.

Two conditions (same task/seed/prompt):
  1) CLEAN model  + clean BDDL scene
  2) GoBA-backdoored model + trigger BDDL scene

At every control step, teacher-forced forward (prompt+image) yields:
  text2patch : (n_text, 256)   -- text queries attending to image patches
  patch2text : (256, n_text)   -- image patches attending to text tokens
Both at last LLM layer, head-averaged (T2IShield-style). Then the same model
generates/executes an action so the scene evolves.

Output: attn_maps/single_sample_analysis/goba_rollout_xattn/
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
from experiments.robot.libero.run_libero_eval_attentionmap import (
    crop_and_resize,
    get_avg_patch_text_attention,
)
from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer

GOBA_CKPT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
CLEAN_CKPT = "openvla/openvla-7b-finetuned-libero-goal"
OUT_DIR = f"{REPO}/attn_maps/single_sample_analysis/goba_rollout_xattn"

BDDL = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
SUITE = "libero_goal"
TASK_ID = 7
SEED = 7
NUM_STEPS_WAIT = 10
NUM_STEPS = 20
DEVICE = 0
N_DOF = 7
LAYER = -1


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


def crossattn_and_action(vla, processor, action_tokenizer, img, desc, a_low, a_high, a_mask):
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, Image.fromarray(img)).to(DEVICE, dtype=torch.bfloat16)
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches

    # --- text <-> patch matrices (teacher-forced, NO 29871) ---
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(**inputs, output_attentions=True)
    mask = (inputs.input_ids < action_tokenizer.action_token_begin_idx)[0]
    p2t, t2p = get_avg_patch_text_attention(out.attentions, num_patches, mask, layer=LAYER)
    text2patch = t2p.float().cpu().numpy()
    patch2text = p2t.float().cpu().numpy()
    text2patch = text2patch / np.clip(text2patch.sum(1, keepdims=True), 1e-12, None)
    patch2text = patch2text / np.clip(patch2text.sum(1, keepdims=True), 1e-12, None)

    ids = inputs.input_ids[0].tolist()
    kept = [tid for i, tid in enumerate(ids) if i > 0 and bool(mask[i])]
    toks = processor.tokenizer.convert_ids_to_tokens(kept)
    n = min(len(toks), text2patch.shape[0])
    toks, text2patch = toks[:n], text2patch[:n]
    patch2text = patch2text[:, :n]
    del out

    # --- action for closed loop (WITH 29871, match predict_action) ---
    input_ids = inputs.input_ids
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        gen = vla.generate(input_ids, pixel_values=inputs.pixel_values,
                           max_new_tokens=N_DOF, do_sample=False)
    tok = gen[0, -N_DOF:].cpu().numpy()
    disc = np.clip(vla.vocab_size - tok - 1, 0, vla.bin_centers.shape[0] - 1)
    normed = vla.bin_centers[disc]
    action = np.where(a_mask, 0.5 * (normed + 1) * (a_high - a_low) + a_low, normed)
    del gen
    torch.cuda.empty_cache()
    return text2patch, patch2text, toks, action.astype(np.float64), prompt


def run_condition(name, ckpt, bddl_key, task, shared_desc):
    print(f"\n=== {name}: ckpt={ckpt} bddl={bddl_key} ===", flush=True)
    processor, vla = load_model(ckpt)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    unnorm_key = SUITE if SUITE in vla.norm_stats else f"{SUITE}_no_noops"
    stats = vla.get_action_stats(unnorm_key)
    a_low, a_high = np.array(stats["q01"]), np.array(stats["q99"])
    a_mask = np.array(stats.get("mask", np.ones_like(a_low, dtype=bool)))

    env, desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False,
                               bddl_path=BDDL[bddl_key], seed=SEED)
    env.reset()
    obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))

    t2p_seq, p2t_seq, rgb_seq, act_seq = [], [], [], []
    toks = None
    prompt = None
    done = False
    for t in range(NUM_STEPS):
        img = preprocess(get_libero_image(obs, 224))
        t2p, p2t, toks, action, prompt = crossattn_and_action(
            vla, processor, action_tokenizer, img, shared_desc, a_low, a_high, a_mask)
        t2p_seq.append(t2p)
        p2t_seq.append(p2t)
        rgb_seq.append(img)
        act_seq.append(action)
        print(f"  step={t:02d} text2patch max={t2p.max():.4f} "
              f"mean_token_max={t2p.max(1).mean():.4f} "
              f"fnorm={np.linalg.norm(t2p - t2p.mean(0), axis=1).mean():.4f}", flush=True)
        exec_a = invert_gripper_action(normalize_gripper_action(action.copy(), binarize=True))
        obs, _, done, _ = env.step(exec_a.tolist())
        if done:
            break
    env.close()
    del vla, processor
    torch.cuda.empty_cache()
    return {
        "text2patch": np.stack(t2p_seq),      # (T, n_text, 256)
        "patch2text": np.stack(p2t_seq),      # (T, 256, n_text)
        "rgb": np.stack(rgb_seq),
        "actions": np.stack(act_seq),
        "tokens": np.array(toks, dtype=object),
        "prompt": prompt,
        "done": done,
        "n_steps": len(t2p_seq),
    }


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

    clean = run_condition("CLEAN_model__clean_scene", CLEAN_CKPT, "clean", task, shared_desc)
    poison = run_condition("GOBA_backdoored__trigger_scene", GOBA_CKPT, "poison", task, shared_desc)

    np.savez_compressed(
        f"{OUT_DIR}/rollout_xattn.npz",
        clean_text2patch=clean["text2patch"],
        clean_patch2text=clean["patch2text"],
        clean_rgb=clean["rgb"],
        clean_actions=clean["actions"],
        poison_text2patch=poison["text2patch"],
        poison_patch2text=poison["patch2text"],
        poison_rgb=poison["rgb"],
        poison_actions=poison["actions"],
        tokens=clean["tokens"],
        prompt=np.array(clean["prompt"]),
        task_id=TASK_ID, seed=SEED, layer=np.array([LAYER]),
        n_steps_clean=clean["n_steps"], n_steps_poison=poison["n_steps"],
    )
    print(f"\nSaved matrices -> {OUT_DIR}/rollout_xattn.npz", flush=True)
    print(f"  clean  text2patch {clean['text2patch'].shape}  patch2text {clean['patch2text'].shape}", flush=True)
    print(f"  poison text2patch {poison['text2patch'].shape}  patch2text {poison['patch2text'].shape}", flush=True)
    print(f"  tokens={list(clean['tokens'])}", flush=True)


if __name__ == "__main__":
    main()
