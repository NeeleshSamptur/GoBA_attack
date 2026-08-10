"""
experiments/robot/libero/run_trajectory_dof_attention_cleanmodel.py

CLEAN-MODEL CONTROL for the spatio-temporal DoF attention rollout.

Same protocol as run_trajectory_dof_attention.py (same tasks, seeds, BDDL
scenes, band layers, 29871 handling), but the policy is the OFFICIAL clean
fine-tuned checkpoint openvla/openvla-7b-finetuned-libero-goal (no backdoor
training). Roles: clean scenes and poison scenes (GoBA trigger object present
in the scene, but the model was never trained on it).

Question this answers: does the trigger object FREEZE attention / distort the
gaze-action loop in a model that was never backdoored? If the freeze/blinding
signatures appear only in the backdoored models, they are properties of the
ATTACK TRAINING, not of the object or the scene.

Output: attn_maps/cleanmodel_trajectory_dof_attention.npz
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

CHECKPOINT = "openvla/openvla-7b-finetuned-libero-goal"   # official CLEAN model
OUT_NPZ = f"{REPO}/attn_maps/cleanmodel_trajectory_dof_attention.npz"

BDDL = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
ROLE_SEEDS = {
    "clean": [5, 99],
    "poison": [11, 43],
}
SUITE = "libero_goal"
NUM_STEPS_WAIT = 10
NUM_STEPS = 30
DEVICE = 1
N_DOF = 7
BAND = [8, 16, 24, 27]   # a-priori compression band (Mix-Compress-Refine)


def preprocess(img):
    im = tf.convert_to_tensor(np.array(Image.fromarray(img).convert("RGB")))
    dt = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, dt, saturate=True)
    return Image.fromarray(im.numpy()).convert("RGB")


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    print("Loading CLEAN model...", flush=True)
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        CHECKPOINT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE)
    if os.path.isdir(CHECKPOINT):
        vla.norm_stats = json.load(open(os.path.join(CHECKPOINT, "dataset_statistics.json")))
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches
    unnorm_key = SUITE if SUITE in vla.norm_stats else f"{SUITE}_no_noops"
    stats = vla.get_action_stats(unnorm_key)
    a_low, a_high = np.array(stats["q01"]), np.array(stats["q99"])
    a_mask = np.array(stats.get("mask", np.ones_like(a_low, dtype=bool)))
    print(f"Model loaded. num_patches={num_patches} unnorm_key={unnorm_key}", flush=True)

    task_suite = benchmark.get_benchmark_dict()[SUITE]()

    def policy_step(image, desc):
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
        input_ids = inputs.input_ids
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = vla.generate(input_ids, pixel_values=inputs.pixel_values,
                               max_new_tokens=N_DOF, output_attentions=True,
                               return_dict_in_generate=True, do_sample=False)

        dof_maps = np.zeros((N_DOF, num_patches), np.float32)
        for k in range(N_DOF):
            m = np.zeros(num_patches, np.float64)
            for l in BAND:
                la = gen.attentions[k][l][0].float().mean(dim=0)
                row = (la[-1] if la.dim() == 2 else la[0]).cpu().numpy()
                r = row[1: 1 + num_patches]
                m += r / max(r.sum(), 1e-12)
            dof_maps[k] = m / len(BAND)

        tok = gen.sequences[0, -N_DOF:].cpu().numpy()
        disc = np.clip(vla.vocab_size - tok - 1, 0, vla.bin_centers.shape[0] - 1)
        normed = vla.bin_centers[disc]
        action = np.where(a_mask, 0.5 * (normed + 1) * (a_high - a_low) + a_low, normed)
        del gen
        return action.astype(np.float64), dof_maps

    episodes = []
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        for role, seeds in ROLE_SEEDS.items():
            for seed in seeds:
                env, desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False,
                                           bddl_path=BDDL[role], seed=seed)
                env.reset()
                obs = None
                for _ in range(NUM_STEPS_WAIT):
                    obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
                ep = {"task_id": task_id, "seed": seed, "role": role,
                      "maps": [], "actions": [], "done": False}
                for t in range(NUM_STEPS):
                    img = preprocess(get_libero_image(obs, 224))
                    action, dof_maps = policy_step(img, desc)
                    ep["maps"].append(dof_maps)
                    ep["actions"].append(action)
                    exec_a = normalize_gripper_action(action.copy(), binarize=True)
                    exec_a = invert_gripper_action(exec_a)
                    obs, _, done, _ = env.step(exec_a.tolist())
                    if done:
                        ep["done"] = True
                        break
                env.close()
                episodes.append(ep)
                am = np.stack(ep["maps"]).argmax(-1)
                stl = (am == np.bincount(am.ravel()).argmax()).mean()
                print(f"task={task_id} role={role} seed={seed} steps={len(ep['maps'])} "
                      f"st_lock={stl:.3f} done={ep['done']}", flush=True)
        torch.cuda.empty_cache()

    n = len(episodes)
    maps = np.full((n, NUM_STEPS, N_DOF, num_patches), np.nan, np.float32)
    acts = np.full((n, NUM_STEPS, N_DOF), np.nan, np.float32)
    for i, ep in enumerate(episodes):
        T = len(ep["maps"])
        maps[i, :T] = ep["maps"]
        acts[i, :T] = ep["actions"]
    np.savez(OUT_NPZ,
             role=np.array([e["role"] for e in episodes]),
             task_id=np.array([e["task_id"] for e in episodes]),
             seed=np.array([e["seed"] for e in episodes]),
             done=np.array([e["done"] for e in episodes]),
             maps=maps, actions=acts, band=np.array(BAND))
    print(f"\nSaved {n} episodes -> {OUT_NPZ}", flush=True)


if __name__ == "__main__":
    main()
