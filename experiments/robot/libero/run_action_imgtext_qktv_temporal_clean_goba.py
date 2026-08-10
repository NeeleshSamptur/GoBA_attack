"""
Clean OpenVLA (GoBA protocol): action→(image+text) QKTV across T steps.
Same BDDL clean/poison scenes as GoBA temporal for direct comparison.
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

CHECKPOINT = "openvla/openvla-7b-finetuned-libero-goal"
OUT = f"{REPO}/attn_maps/action_imgtext_qktv_temporal"
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
LAYER = -1
# limit tasks via env for smoke tests
MAX_TASKS = int(os.environ.get("MAX_TASKS", "10"))


def preprocess(img):
    im = tf.convert_to_tensor(np.array(Image.fromarray(img).convert("RGB")))
    dt = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, dt, saturate=True)
    return np.array(Image.fromarray(im.numpy()).convert("RGB"))


def ftt(P):
    P = np.clip(P, 0, None)
    P = P / np.clip(P.sum(-1, keepdims=True), 1e-12, None)
    return float(np.linalg.norm(P - P.mean(0), axis=1).mean())


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    os.makedirs(OUT, exist_ok=True)

    print("Loading CLEAN OpenVLA (GoBA protocol)...", flush=True)
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        CHECKPOINT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE)
    vla.eval()
    if os.path.isdir(CHECKPOINT) and os.path.exists(os.path.join(CHECKPOINT, "dataset_statistics.json")):
        vla.norm_stats = json.load(open(os.path.join(CHECKPOINT, "dataset_statistics.json")))
    elif not getattr(vla, "norm_stats", None):
        from huggingface_hub import hf_hub_download
        stats_path = hf_hub_download(CHECKPOINT, "dataset_statistics.json")
        vla.norm_stats = json.load(open(stats_path))
    n_patch = vla.vision_backbone.featurizer.patch_embed.num_patches
    unnorm_key = SUITE if SUITE in vla.norm_stats else f"{SUITE}_no_noops"
    stats = vla.get_action_stats(unnorm_key)
    a_low, a_high = np.array(stats["q01"]), np.array(stats["q99"])
    a_mask = np.array(stats.get("mask", np.ones_like(a_low, dtype=bool)))

    llm = vla.language_model
    layers = llm.model.layers
    n_layers = len(layers)
    attn0 = layers[0].self_attn
    n_heads, head_dim = attn0.num_heads, attn0.head_dim
    hidden = n_heads * head_dim
    v_cache = {}
    handles = [lyr.self_attn.v_proj.register_forward_hook(
        lambda _m, _i, out, l=l: v_cache.__setitem__(l, out))
        for l, lyr in enumerate(layers)]
    Wo = {l: layers[l].self_attn.o_proj.weight.detach().view(hidden, n_heads, head_dim)
          for l in range(n_layers)}
    print(f"patches={n_patch} layers={n_layers}", flush=True)

    task_suite = benchmark.get_benchmark_dict()[SUITE]()
    episodes = []

    def step_maps(img, desc):
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        inputs = processor(prompt, Image.fromarray(img)).to(DEVICE, dtype=torch.bfloat16)
        input_ids = inputs.input_ids
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)
        n_lang = input_ids.shape[-1] - 1
        t1 = 1 + n_patch + n_lang
        k_cols = list(range(1, t1))

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

        v_cache.clear()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = vla(input_ids=seq, pixel_values=inputs.pixel_values,
                      output_attentions=True, return_dict=True)
        N = out.attentions[0].shape[-1]
        q_rows = list(range(N - N_DOF, N))
        l = n_layers + LAYER if LAYER < 0 else LAYER
        A = out.attentions[l][0].float()
        vh = v_cache[l][0].float().view(N, n_heads, head_dim)
        f = torch.einsum("nhd,ohd->nho", vh, Wo[l].float())
        f_norms = f.norm(dim=-1)
        Aq = A[:, q_rows, :]
        w = Aq.mean(0)[:, k_cols].cpu().numpy()
        n = torch.einsum("hqn,nh->qn", Aq, f_norms)[:, k_cols].cpu().numpy()
        w = w / np.clip(w.sum(1, keepdims=True), 1e-12, None)
        n = n / np.clip(n.sum(1, keepdims=True), 1e-12, None)
        # image-only portion for spatial maps
        w_img = w[:, :n_patch]
        n_img = n[:, :n_patch]
        w_img = w_img / np.clip(w_img.sum(1, keepdims=True), 1e-12, None)
        n_img = n_img / np.clip(n_img.sum(1, keepdims=True), 1e-12, None)
        del out, A, vh, f, f_norms
        torch.cuda.empty_cache()
        return action.astype(np.float64), dict(
            ftt_w=ftt(w), ftt_qktv=ftt(n),
            mean_w_img=w_img.mean(0).astype(np.float32),
            mean_qktv_img=n_img.mean(0).astype(np.float32),
            n_lang=n_lang,
        )

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
                      ftt_w=[], ftt_qktv=[], mean_w_img=[], mean_qktv_img=[], rgb=[], n_lang=[])
            for t in range(NUM_STEPS):
                img = preprocess(get_libero_image(obs, 224))
                action, m = step_maps(img, desc)
                ep["ftt_w"].append(m["ftt_w"])
                ep["ftt_qktv"].append(m["ftt_qktv"])
                ep["mean_w_img"].append(m["mean_w_img"])
                ep["mean_qktv_img"].append(m["mean_qktv_img"])
                ep["rgb"].append(img)
                ep["n_lang"].append(m["n_lang"])
                exec_a = invert_gripper_action(normalize_gripper_action(action.copy(), binarize=True))
                obs, _, done, _ = env.step(exec_a.tolist())
                if done:
                    break
            env.close()
            T = len(ep["ftt_w"])
            print(f"CLEAN_goba-proto task={task_id} {cond:6s} T={T}  "
                  f"FTT_qktv t0={ep['ftt_qktv'][0]:.4f} mean={np.mean(ep['ftt_qktv']):.4f}", flush=True)
            episodes.append(ep)

    for h in handles:
        h.remove()

    n = len(episodes)
    Tmax = NUM_STEPS
    ftt_w = np.full((n, Tmax), np.nan)
    ftt_q = np.full((n, Tmax), np.nan)
    maps_w = np.full((n, Tmax, n_patch), np.nan, np.float32)
    maps_q = np.full((n, Tmax, n_patch), np.nan, np.float32)
    rgb = np.zeros((n, Tmax, 224, 224, 3), np.uint8)
    for i, ep in enumerate(episodes):
        T = len(ep["ftt_w"])
        ftt_w[i, :T] = ep["ftt_w"]
        ftt_q[i, :T] = ep["ftt_qktv"]
        maps_w[i, :T] = ep["mean_w_img"]
        maps_q[i, :T] = ep["mean_qktv_img"]
        for t in range(T):
            rgb[i, t] = ep["rgb"][t]
    np.savez_compressed(
        f"{OUT}/clean_goba_protocol_temporal.npz",
        task_id=np.array([e["task_id"] for e in episodes]),
        cond=np.array([e["cond"] for e in episodes]),
        seed=np.array([e["seed"] for e in episodes]),
        ftt_weight=ftt_w, ftt_qktv=ftt_q,
        maps_weight=maps_w, maps_qktv=maps_q, rgb=rgb,
        n_patch=n_patch, grid=16,
    )
    print(f"Saved {n} episodes -> {OUT}/clean_goba_protocol_temporal.npz", flush=True)


if __name__ == "__main__":
    main()
