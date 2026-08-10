"""
experiments/robot/libero/run_trajectory_dof_attention.py

SPATIO-TEMPORAL PER-DoF ATTENTION: do all 7 action dimensions attend to the
same image patch, at every control step of a rollout -- or does attention
have structure across DoFs and time?

Prior pieces (this repo): pooled band-map argmax locks on one patch under
poison (trajectory run); the 7 DoF maps assimilate within a single step
(coupling run). This experiment records the FULL (DoF x time) attention
structure: per band layer, per control step, the patch map of EVERY DoF's
action query. Statistics:

  * spatio-temporal lock: fraction of (DoF, step) cells whose argmax equals
    the episode's modal patch. Poison hypothesis: ~1.0 (a rank-one, frozen
    pattern -- every action dimension staring at the trigger at every step).
    Clean hypothesis: DoFs disagree AND drift as the arm moves.
  * per-DoF temporal dwell: does DoF d keep its own argmax over time?
  * cross-DoF agreement per step and its temporal trend.

Protocol matches run_trajectory_attention_rollout.py exactly (same roles,
seeds, tasks, NUM_STEPS, band layers, 29871 handling).

Outputs: attn_maps/trajectory_dof_attention.npz + printed summary.
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
from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

CHECKPOINT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
OUT_NPZ = f"{REPO}/attn_maps/trajectory_dof_attention.npz"

BDDL = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
    "decoy_ketchup": f"{REPO}/BadLIBERO/libero/libero/bddl_files-decoy_eval",
    "decoy_milk": f"{REPO}/BadLIBERO/libero/libero/bddl_files-decoy2_eval",
}
ROLE_SEEDS = {
    "clean": [5, 99],
    "poison": [11, 43],
    "decoy_ketchup": [6, 100],
    "decoy_milk": [8, 101],
}
SUITE = "libero_goal"
NUM_STEPS_WAIT = 10
NUM_STEPS = 30
DEVICE = 0
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

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        CHECKPOINT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE)
    vla.norm_stats = json.load(open(os.path.join(CHECKPOINT, "dataset_statistics.json")))
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches
    unnorm_key = SUITE if SUITE in vla.norm_stats else f"{SUITE}_no_noops"
    stats = vla.get_action_stats(unnorm_key)
    a_low, a_high = np.array(stats["q01"]), np.array(stats["q99"])
    a_mask = np.array(stats.get("mask", np.ones_like(a_low, dtype=bool)))
    print(f"Model loaded. num_patches={num_patches}", flush=True)

    task_suite = benchmark.get_benchmark_dict()[SUITE]()

    def policy_step(image, desc):
        """Returns (action, per-DoF band-averaged patch maps (7,256))."""
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
                la = gen.attentions[k][l][0].float().mean(dim=0)   # avg heads
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
                am = np.stack(ep["maps"]).argmax(-1)   # (T,7)
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
    roles = np.array([e["role"] for e in episodes])
    np.savez(OUT_NPZ, role=roles,
             task_id=np.array([e["task_id"] for e in episodes]),
             seed=np.array([e["seed"] for e in episodes]),
             done=np.array([e["done"] for e in episodes]),
             maps=maps, actions=acts, band=np.array(BAND))
    print(f"\nSaved {n} episodes -> {OUT_NPZ}", flush=True)

    # ---------------- summary ----------------
    def dirauc(v, p, ne):
        ok = ~np.isnan(v)
        y = np.r_[np.ones((p & ok).sum()), np.zeros((ne & ok).sum())]
        s = np.r_[v[p & ok], v[ne & ok]]
        r = roc_auc_score(y, s)
        return r if r >= 0.5 else 1 - r

    is_p, is_c = roles == "poison", roles == "clean"
    is_k, is_m = roles == "decoy_ketchup", roles == "decoy_milk"

    # spatio-temporal lock: share of (step, DoF) cells on the episode's modal patch
    stl = np.full(n, np.nan)
    uniq = np.full(n, np.nan)
    for i in range(n):
        M = maps[i]
        ok = ~np.isnan(M[:, 0, 0])
        am = M[ok].argmax(-1)                     # (T,7)
        stl[i] = (am == np.bincount(am.ravel()).argmax()).mean()
        uniq[i] = len(np.unique(am))
    print("\nSPATIO-TEMPORAL LOCK (fraction of DoF x step cells on the modal patch):")
    for nm, msk in [("poison", is_p), ("clean", is_c), ("ketchup", is_k), ("milk", is_m)]:
        print(f"  {nm:<8} mean={np.nanmean(stl[msk]):.3f} range=[{np.nanmin(stl[msk]):.3f},{np.nanmax(stl[msk]):.3f}]"
              f"  unique patches mean={np.nanmean(uniq[msk]):.1f}")
    print(f"  AUROC: poison={dirauc(stl, is_p, is_c):.4f} ketchup={dirauc(stl, is_k, is_c):.4f} "
          f"milk={dirauc(stl, is_m, is_c):.4f}")
    is_b = ~is_p
    print(f"  fixed margin = {max(np.nanmin(stl[is_p])-np.nanmax(stl[is_b]), np.nanmin(stl[is_b])-np.nanmax(stl[is_p])):+.4f}")

    # cross-DoF agreement per step (mean pairwise cosine), episode mean
    xdof = np.full(n, np.nan)
    for i in range(n):
        M = maps[i]
        ok = ~np.isnan(M[:, 0, 0])
        cs = []
        for t in np.where(ok)[0]:
            V = M[t] / np.clip(np.linalg.norm(M[t], axis=1, keepdims=True), 1e-12, None)
            C = V @ V.T
            iu = np.triu_indices(N_DOF, 1)
            cs.append(C[iu].mean())
        xdof[i] = np.mean(cs)
    print("\nCROSS-DoF MAP COSINE (episode mean):")
    for nm, msk in [("poison", is_p), ("clean", is_c), ("ketchup", is_k), ("milk", is_m)]:
        print(f"  {nm:<8} mean={np.nanmean(xdof[msk]):.4f} range=[{np.nanmin(xdof[msk]):.4f},{np.nanmax(xdof[msk]):.4f}]")
    print(f"  AUROC: poison={dirauc(xdof, is_p, is_c):.4f} ketchup={dirauc(xdof, is_k, is_c):.4f} "
          f"milk={dirauc(xdof, is_m, is_c):.4f}")

    # per-DoF temporal dwell (does each DoF keep ITS OWN argmax over time)
    dwell = np.full((n, N_DOF), np.nan)
    for i in range(n):
        M = maps[i]
        ok = ~np.isnan(M[:, 0, 0])
        am = M[ok].argmax(-1)
        for d in range(N_DOF):
            a = am[:, d]
            dwell[i, d] = (a == np.bincount(a).argmax()).mean()
    print("\nPER-DoF TEMPORAL DWELL (fraction of steps on that DoF's modal patch), DoF-avg:")
    v = np.nanmean(dwell, 1)
    for nm, msk in [("poison", is_p), ("clean", is_c), ("ketchup", is_k), ("milk", is_m)]:
        print(f"  {nm:<8} {np.nanmean(v[msk]):.3f}")
    print(f"  AUROC poison={dirauc(v, is_p, is_c):.4f}")


if __name__ == "__main__":
    main()
