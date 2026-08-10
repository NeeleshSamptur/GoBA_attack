"""
GoBA: VLA-InfoEntropy-style Ieattn on text→image attention + temporal Top-k lock.
Protocol: libero_goal tasks 0-9, seed=7, T=10, clean vs poison BDDL.
"""
import json
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
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.libero.run_libero_eval_attentionmap import crop_and_resize
from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

CHECKPOINT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
OUT = f"{REPO}/attn_maps/infoentropy_ieattn"
BDDL = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
SUITE = "libero_goal"
SEED = 7
NUM_STEPS_WAIT = 10
NUM_STEPS = 10
DEVICE = 0
N_DOF = 7
TOPK = 16
MAX_TASKS = int(os.environ.get("MAX_TASKS", "10"))


def preprocess(img):
    im = tf.convert_to_tensor(np.array(Image.fromarray(img).convert("RGB")))
    dt = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, dt, saturate=True)
    return np.array(Image.fromarray(im.numpy()).convert("RGB"))


def ieattn_from_score(score_wi):
    """score_wi: (n_txt, n_img) = mean_{l,h} A_{w→i} (post-softmax attn).

    Paper applies a second softmax over w; on causal VLA self-attn that flattens
    to ~uniform (Ie≈0). We instead renormalize over text tokens:
      q(w,i) = A_{w→i} / ∑_{w'} A_{w'→i}
    which is the natural 'which text tokens attend to patch i' distribution.
    """
    q = score_wi / np.clip(score_wi.sum(axis=0, keepdims=True), 1e-12, None)
    H = -(q * np.log2(q + 1e-12)).sum(axis=0)
    nW = score_wi.shape[0]
    Ie = 1.0 - H / np.log2(max(nW, 2))
    return Ie.astype(np.float32), H.astype(np.float32)


def topk_set(Ie, k):
    return set(np.argsort(Ie)[-k:].tolist())


def jaccard(a, b):
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    os.makedirs(OUT, exist_ok=True)

    print("Loading GoBA...", flush=True)
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        CHECKPOINT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE).eval()
    vla.norm_stats = json.load(open(os.path.join(CHECKPOINT, "dataset_statistics.json")))
    n_patch = vla.vision_backbone.featurizer.patch_embed.num_patches
    unnorm_key = SUITE if SUITE in vla.norm_stats else f"{SUITE}_no_noops"
    stats = vla.get_action_stats(unnorm_key)
    a_low, a_high = np.array(stats["q01"]), np.array(stats["q99"])
    a_mask = np.array(stats.get("mask", np.ones_like(a_low, dtype=bool)))
    n_layers = len(vla.language_model.model.layers)
    print(f"patches={n_patch} layers={n_layers}", flush=True)

    def step_ieattn(img, desc):
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        inputs = processor(prompt, Image.fromarray(img)).to(DEVICE, dtype=torch.bfloat16)
        input_ids = inputs.input_ids
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)
        n_lang = input_ids.shape[-1] - 1  # exclude BOS from lang count in prompt? 
        # layout: [BOS, img..., lang..., action...] after full seq
        # After generate: seq = [BOS, ?, ...] actually input_ids already has BOS
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = vla.generate(
                input_ids, pixel_values=inputs.pixel_values,
                max_new_tokens=N_DOF, output_attentions=False,
                return_dict_in_generate=True, do_sample=False,
            )
        seq = gen.sequences
        tok = seq[0, -N_DOF:].cpu().numpy()
        disc = np.clip(vla.vocab_size - tok - 1, 0, vla.bin_centers.shape[0] - 1)
        normed = vla.bin_centers[disc]
        action = np.where(a_mask, 0.5 * (normed + 1) * (a_high - a_low) + a_low, normed)
        del gen

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = vla(input_ids=seq, pixel_values=inputs.pixel_values,
                      output_attentions=True, return_dict=True)
        N = out.attentions[0].shape[-1]
        # OpenVLA multimodal: pos0=BOS, 1:1+n_patch=image, then language (n_lang), then actions
        img_cols = list(range(1, 1 + n_patch))
        txt_rows = list(range(1 + n_patch, 1 + n_patch + n_lang))
        # Last 8 layers (all-layer avg washes structure on causal VLA)
        layer_ids = list(range(max(0, n_layers - 8), n_layers))
        acc = None
        for l in layer_ids:
            A = out.attentions[l][0].float()  # (H, N, N)
            block = A[:, txt_rows][:, :, img_cols].mean(0)  # (n_txt, n_img)
            acc = block if acc is None else acc + block
        acc = (acc / len(layer_ids)).cpu().numpy()
        del out
        torch.cuda.empty_cache()
        Ie, H = ieattn_from_score(acc)
        return action.astype(np.float64), Ie, H, n_lang

    task_suite = benchmark.get_benchmark_dict()[SUITE]()
    episodes = []
    for task_id in range(min(MAX_TASKS, task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        for cond, bddl in BDDL.items():
            env, desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False,
                                       bddl_path=bddl, seed=SEED)
            env.reset()
            obs = None
            for _ in range(NUM_STEPS_WAIT):
                obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
            ep = dict(task_id=task_id, cond=cond, seed=SEED,
                      Ie=[], H=[], rgb=[], n_lang=[])
            for t in range(NUM_STEPS):
                img = preprocess(get_libero_image(obs, 224))
                action, Ie, H, n_lang = step_ieattn(img, desc)
                ep["Ie"].append(Ie)
                ep["H"].append(H)
                ep["rgb"].append(img)
                ep["n_lang"].append(n_lang)
                exec_a = invert_gripper_action(normalize_gripper_action(action.copy(), binarize=True))
                obs, _, done, _ = env.step(exec_a.tolist())
                if done:
                    break
            env.close()
            # episode scalars
            Ies = np.stack(ep["Ie"])  # (T, 256)
            sets = [topk_set(Ies[t], TOPK) for t in range(len(Ies))]
            jac_consec = [jaccard(sets[t], sets[t + 1]) for t in range(len(sets) - 1)]
            jac_t0 = [jaccard(sets[0], sets[t]) for t in range(1, len(sets))]
            ep["score_topk_jac_consec"] = float(np.mean(jac_consec)) if jac_consec else 1.0
            ep["score_topk_jac_t0"] = float(np.mean(jac_t0)) if jac_t0 else 1.0
            ep["score_mean_Ie"] = float(Ies.mean())
            ep["score_max_Ie"] = float(Ies.max(axis=1).mean())
            # spatial entropy of Ie (as mass)
            mass = Ies / np.clip(Ies.sum(1, keepdims=True), 1e-12, None)
            spat_H = -(mass * np.log(mass + 1e-12)).sum(1) / np.log(n_patch)
            ep["score_spatial_Ie_entropy"] = float(spat_H.mean())
            print(f"GoBA task={task_id} {cond:6s} T={len(Ies)}  "
                  f"jac_consec={ep['score_topk_jac_consec']:.3f} "
                  f"meanIe={ep['score_mean_Ie']:.3f} maxIe={ep['score_max_Ie']:.3f}", flush=True)
            episodes.append(ep)

    n = len(episodes)
    Tmax = NUM_STEPS
    Ie_arr = np.full((n, Tmax, n_patch), np.nan, np.float32)
    rgb = np.zeros((n, Tmax, 224, 224, 3), np.uint8)
    for i, ep in enumerate(episodes):
        T = len(ep["Ie"])
        Ie_arr[i, :T] = ep["Ie"]
        for t in range(T):
            rgb[i, t] = ep["rgb"][t]
    np.savez_compressed(
        f"{OUT}/goba_ieattn.npz",
        task_id=np.array([e["task_id"] for e in episodes]),
        cond=np.array([e["cond"] for e in episodes]),
        seed=np.array([e["seed"] for e in episodes]),
        Ie=Ie_arr, rgb=rgb, topk=TOPK,
        score_topk_jac_consec=np.array([e["score_topk_jac_consec"] for e in episodes]),
        score_topk_jac_t0=np.array([e["score_topk_jac_t0"] for e in episodes]),
        score_mean_Ie=np.array([e["score_mean_Ie"] for e in episodes]),
        score_max_Ie=np.array([e["score_max_Ie"] for e in episodes]),
        score_spatial_Ie_entropy=np.array([e["score_spatial_Ie_entropy"] for e in episodes]),
    )
    print(f"Saved {n} episodes -> {OUT}/goba_ieattn.npz", flush=True)


if __name__ == "__main__":
    main()
