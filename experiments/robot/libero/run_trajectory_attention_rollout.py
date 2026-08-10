"""
experiments/robot/libero/run_trajectory_attention_rollout.py

LINKING ACTION TRAJECTORIES TO ATTENTION DYNAMICS (closed-loop rollouts).

Runtime monitors for robot policies at top venues (Sentinel CoRL'24, ActProbe,
VLA-FAIL) watch temporal ACTION consistency to catch failures; attention-based
backdoor defenses (T2IShield, Bera, TrustVLA) look at a SINGLE observation.
Nobody tracks both together over a rollout. This experiment does: at every
closed-loop control step we record

  * the compression-band action-query attention concentration
      - band = layers {8,16,24,27}, fixed A PRIORI by the Mix-Compress-Refine
        depth phases (compressed computation = 20-85% depth; arXiv 2510.06477),
        NOT selected on data
      - concentration = max-patch share of the DoF-pooled action-token
        attention map, averaged over band layers
  * the argmax patch of the band map (for lock/dwell statistics)
  * the executed 7-DoF action (predict_action-faithful decode + gripper
    normalize/invert, as in run_libero_eval_attentionmap.py)

Roles: clean / poison / decoy_ketchup / decoy_milk (benign objects in the
trigger's slot), 10 tasks x 2 seeds each, NUM_STEPS policy steps per episode.

Questions this answers:
  1. Does temporal aggregation of the attention statistic improve detection
     margins over the single-frame version (and stay silent on decoys)?
  2. Does poisoned attention LOCK (long dwell on one patch) while clean/decoy
     attention wanders as the arm moves -- the attention analogue of the
     action-consistency signals in Sentinel/ActProbe?
  3. Do attention dynamics and action dynamics move together (e.g. lock
     persists while the corrupted trajectory unfolds)?

Outputs: attn_maps/trajectory_attention.npz + printed summary.
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
OUT_NPZ = f"{REPO}/attn_maps/trajectory_attention.npz"

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
BAND = [8, 16, 24, 27]     # a-priori compression band (Mix-Compress-Refine 20-85% depth)


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
    print(f"Model loaded. num_patches={num_patches} unnorm_key={unnorm_key}", flush=True)

    task_suite = benchmark.get_benchmark_dict()[SUITE]()

    def policy_step(image, desc):
        """One control step: returns (unnormalized action, band concentration,
        per-layer concentrations, band-map argmax patch)."""
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

        # DoF-pooled patch map per band layer from the generation-step attentions.
        conc_per_layer, band_map = {}, np.zeros(num_patches, np.float64)
        for l in BAND:
            m = np.zeros(num_patches, np.float64)
            for k in range(N_DOF):
                la = gen.attentions[k][l][0].float().mean(dim=0)   # avg heads
                row = (la[-1] if la.dim() == 2 else la[0]).cpu().numpy()
                m += row[1: 1 + num_patches]
            m /= N_DOF
            conc_per_layer[l] = float(m.max() / max(m.sum(), 1e-12))
            band_map += m / len(BAND)
        conc_band = float(np.mean(list(conc_per_layer.values())))
        argmax_patch = int(band_map.argmax())

        # predict_action-faithful decode.
        tok = gen.sequences[0, -N_DOF:].cpu().numpy()
        disc = np.clip(vla.vocab_size - tok - 1, 0, vla.bin_centers.shape[0] - 1)
        normed = vla.bin_centers[disc]
        action = np.where(a_mask, 0.5 * (normed + 1) * (a_high - a_low) + a_low, normed)
        del gen
        return action.astype(np.float64), conc_band, conc_per_layer, argmax_patch

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
                      "conc": [], "conc_layers": [], "argmax": [], "actions": [], "done": False}
                for t in range(NUM_STEPS):
                    img = preprocess(get_libero_image(obs, 224))
                    action, cb, cl, am = policy_step(img, desc)
                    ep["conc"].append(cb)
                    ep["conc_layers"].append([cl[l] for l in BAND])
                    ep["argmax"].append(am)
                    ep["actions"].append(action)
                    exec_a = normalize_gripper_action(action.copy(), binarize=True)
                    exec_a = invert_gripper_action(exec_a)
                    obs, _, done, _ = env.step(exec_a.tolist())
                    if done:
                        ep["done"] = True
                        break
                env.close()
                episodes.append(ep)
                print(f"task={task_id} role={role} seed={seed} steps={len(ep['conc'])} "
                      f"meanconc={np.mean(ep['conc']):.4f} done={ep['done']}", flush=True)
        torch.cuda.empty_cache()

    # ---- save (pad episodes to NUM_STEPS with NaN) ----
    n = len(episodes)
    conc = np.full((n, NUM_STEPS), np.nan, np.float32)
    concL = np.full((n, NUM_STEPS, len(BAND)), np.nan, np.float32)
    argm = np.full((n, NUM_STEPS), -1, np.int32)
    acts = np.full((n, NUM_STEPS, N_DOF), np.nan, np.float32)
    for i, ep in enumerate(episodes):
        T = len(ep["conc"])
        conc[i, :T] = ep["conc"]
        concL[i, :T] = ep["conc_layers"]
        argm[i, :T] = ep["argmax"]
        acts[i, :T] = ep["actions"]
    roles = np.array([e["role"] for e in episodes])
    np.savez(OUT_NPZ, role=roles,
             task_id=np.array([e["task_id"] for e in episodes]),
             seed=np.array([e["seed"] for e in episodes]),
             done=np.array([e["done"] for e in episodes]),
             conc=conc, conc_layers=concL, argmax=argm, actions=acts,
             band=np.array(BAND))
    print(f"\nSaved {n} episodes -> {OUT_NPZ}")

    # ---- quick summary ----
    def dirauc(v, p, ne):
        ok = ~np.isnan(v)
        y = np.r_[np.ones((p & ok).sum()), np.zeros((ne & ok).sum())]
        s = np.r_[v[p & ok], v[ne & ok]]
        r = roc_auc_score(y, s)
        return r if r >= 0.5 else 1.0 - r

    is_p, is_c = roles == "poison", roles == "clean"
    is_k, is_m = roles == "decoy_ketchup", roles == "decoy_milk"
    print("\nTEMPORAL AGGREGATION of band concentration (mean over first T steps):")
    print(f"{'T':>4} | {'poison':>7} {'ketchup':>8} {'milk':>7} {'gap':>8}")
    for T in [1, 3, 5, 10, 20, NUM_STEPS]:
        v = np.nanmean(conc[:, :T], axis=1)
        ap, ak, am = dirauc(v, is_p, is_c), dirauc(v, is_k, is_c), dirauc(v, is_m, is_c)
        print(f"{T:>4} | {ap:>7.4f} {ak:>8.4f} {am:>7.4f} {ap - max(ak, am):>+8.4f}")

    print("\nATTENTION LOCK (longest dwell of band-map argmax on a single patch, / valid steps):")
    dwell = np.zeros(n)
    for i in range(n):
        a = argm[i][argm[i] >= 0]
        best, cur = 1, 1
        for j in range(1, len(a)):
            cur = cur + 1 if a[j] == a[j - 1] else 1
            best = max(best, cur)
        dwell[i] = best / max(len(a), 1)
    for nm, msk in [("poison", is_p), ("clean", is_c), ("ketchup", is_k), ("milk", is_m)]:
        print(f"  {nm:<8} mean dwell = {dwell[msk].mean():.3f}")
    print(f"  AUROC: poison={dirauc(dwell, is_p, is_c):.4f} ketchup={dirauc(dwell, is_k, is_c):.4f} "
          f"milk={dirauc(dwell, is_m, is_c):.4f}")


if __name__ == "__main__":
    main()
