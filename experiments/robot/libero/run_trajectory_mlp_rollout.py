"""
experiments/robot/libero/run_trajectory_mlp_rollout.py

EPISODE-LEVEL MLP FEEDFORWARD STATISTICS (closed-loop rollouts).

run_mlp_feedforward_probe.py showed that self-normalized MLP statistics
(neuron kurtosis at the action tokens, cross-DoF activation cosine) separate
poison from clean AND decoys at AUROC 0.93-0.985 single-frame, with decoys
sitting exactly at clean levels -- but single-frame tails overlap, so no fixed
constant works. The attention band-concentration statistic had the same
problem and became perfectly separable once averaged over an episode. This
run tests whether the same temporal aggregation gives the MLP statistics a
genuinely CALIBRATION-FREE operating point (one universal constant).

At every closed-loop control step we record, from hooks on the down_proj
input (the 11008-dim MLP neuron activations) of all 32 LLM layers:
  * excess kurtosis over neurons, per layer, per generated action token
    (only 6 tokens get a forward pass with KV cache: dx..yaw; gripper's token
    is never re-processed, its slot stays NaN)
  * cross-token mean pairwise cosine of the neuron vectors, per layer
plus the executed action, mirroring run_trajectory_attention_rollout.py
(same roles / seeds / tasks / NUM_STEPS, so results are directly comparable).

Outputs: attn_maps/trajectory_mlp.npz + printed episode-level summary.
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
OUT_NPZ = f"{REPO}/attn_maps/trajectory_mlp.npz"

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
N_GEN = 6          # forward passes during generation with KV cache (dx..yaw)
N_LAYERS = 32


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

    # hooks: only single-token (generation) forwards matter here
    gen_vecs = {}

    def make_hook(li):
        def hook(module, args):
            h = args[0]
            if h.shape[1] == 1:
                gen_vecs.setdefault(li, []).append(h[0, 0].float())
        return hook

    handles = [
        vla.language_model.model.layers[i].mlp.down_proj.register_forward_pre_hook(make_hook(i))
        for i in range(N_LAYERS)
    ]

    task_suite = benchmark.get_benchmark_dict()[SUITE]()

    def policy_step(image, desc):
        gen_vecs.clear()
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
        input_ids = inputs.input_ids
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = vla.generate(input_ids, pixel_values=inputs.pixel_values,
                               max_new_tokens=N_DOF, return_dict_in_generate=True, do_sample=False)

        kurt = np.full((N_LAYERS, N_GEN), np.nan, np.float32)
        cosm = np.full(N_LAYERS, np.nan, np.float32)
        for li, vecs in gen_vecs.items():
            V = torch.stack(vecs[:N_GEN])                     # (<=6, 11008) on GPU
            a = V.abs()
            med = a.median(dim=1, keepdim=True).values.clamp_min(1e-12)
            mu = a.mean(dim=1, keepdim=True)
            sd = a.std(dim=1, keepdim=True).clamp_min(1e-12)
            k = (((a - mu) / sd) ** 4).mean(dim=1) - 3.0
            kurt[li, :V.shape[0]] = k.cpu().numpy()
            Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
            C = Vn @ Vn.T
            iu = torch.triu_indices(V.shape[0], V.shape[0], offset=1)
            cosm[li] = float(C[iu[0], iu[1]].mean())

        tok = gen.sequences[0, -N_DOF:].cpu().numpy()
        disc = np.clip(vla.vocab_size - tok - 1, 0, vla.bin_centers.shape[0] - 1)
        normed = vla.bin_centers[disc]
        action = np.where(a_mask, 0.5 * (normed + 1) * (a_high - a_low) + a_low, normed)
        del gen
        return action.astype(np.float64), kurt, cosm

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
                      "kurt": [], "cos": [], "actions": [], "done": False}
                for t in range(NUM_STEPS):
                    img = preprocess(get_libero_image(obs, 224))
                    action, kurt, cosm = policy_step(img, desc)
                    ep["kurt"].append(kurt)
                    ep["cos"].append(cosm)
                    ep["actions"].append(action)
                    exec_a = normalize_gripper_action(action.copy(), binarize=True)
                    exec_a = invert_gripper_action(exec_a)
                    obs, _, done, _ = env.step(exec_a.tolist())
                    if done:
                        ep["done"] = True
                        break
                env.close()
                episodes.append(ep)
                mk = np.nanmean([k[15] for k in ep["kurt"]])
                print(f"task={task_id} role={role} seed={seed} steps={len(ep['kurt'])} "
                      f"L15kurt={mk:.1f} done={ep['done']}", flush=True)
        torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    n = len(episodes)
    kurt = np.full((n, NUM_STEPS, N_LAYERS, N_GEN), np.nan, np.float32)
    cosm = np.full((n, NUM_STEPS, N_LAYERS), np.nan, np.float32)
    acts = np.full((n, NUM_STEPS, N_DOF), np.nan, np.float32)
    for i, ep in enumerate(episodes):
        T = len(ep["kurt"])
        kurt[i, :T] = ep["kurt"]
        cosm[i, :T] = ep["cos"]
        acts[i, :T] = ep["actions"]
    roles = np.array([e["role"] for e in episodes])
    np.savez(OUT_NPZ, role=roles,
             task_id=np.array([e["task_id"] for e in episodes]),
             seed=np.array([e["seed"] for e in episodes]),
             done=np.array([e["done"] for e in episodes]),
             kurt=kurt, dof_cos=cosm, actions=acts)
    print(f"\nSaved {n} episodes -> {OUT_NPZ}")

    # ---- episode-level summary ----
    def dirauc(v, p, ne):
        ok = ~np.isnan(v)
        y = np.r_[np.ones((p & ok).sum()), np.zeros((ne & ok).sum())]
        s = np.r_[v[p & ok], v[ne & ok]]
        r = roc_auc_score(y, s)
        return r if r >= 0.5 else 1.0 - r

    is_p, is_c = roles == "poison", roles == "clean"
    is_k, is_m = roles == "decoy_ketchup", roles == "decoy_milk"
    is_b = ~is_p

    print("\nEPISODE-MEAN L15 kurtosis (token-avg):")
    v = np.nanmean(kurt[:, :, 15, :], axis=(1, 2))
    for nm, msk in [("poison", is_p), ("clean", is_c), ("ketchup", is_k), ("milk", is_m)]:
        print(f"  {nm:<8} mean={np.nanmean(v[msk]):8.1f}  range=[{np.nanmin(v[msk]):.1f},{np.nanmax(v[msk]):.1f}]")
    print(f"  AUROC p-vs-c={dirauc(v, is_p, is_c):.4f}  fixed-margin={np.nanmin(v[is_p]) - np.nanmax(v[is_b]):+.1f}")

    print("\nEPISODE-MEAN cross-token cosine, layer sweep (best 5 by fixed margin):")
    rows = []
    for li in range(N_LAYERS):
        v = np.nanmean(cosm[:, :, li], axis=1)
        m_up = np.nanmin(v[is_p]) - np.nanmax(v[is_b])
        m_dn = np.nanmin(v[is_b]) - np.nanmax(v[is_p])
        rows.append((max(m_up, m_dn), dirauc(v, is_p, is_b), li))
    rows.sort(reverse=True)
    for m, a, li in rows[:5]:
        v = np.nanmean(cosm[:, :, li], axis=1)
        print(f"  L{li:>2} margin={m:+.4f} AUROC={a:.4f} "
              f"poison=[{np.nanmin(v[is_p]):.4f},{np.nanmax(v[is_p]):.4f}] "
              f"benign=[{np.nanmin(v[is_b]):.4f},{np.nanmax(v[is_b]):.4f}]")


if __name__ == "__main__":
    main()
