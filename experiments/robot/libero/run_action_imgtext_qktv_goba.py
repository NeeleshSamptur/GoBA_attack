"""
GoBA: one demo per task, action-query × (image+text) keys.

For each of 10 LIBERO-Goal tasks (seed=7):
  clean BDDL scene vs poison BDDL scene, same instruction.
Compute last-layer maps with action tokens as queries and the concatenated
[image patches | instruction tokens] as keys, two ways:

  weight   head-avg α                          (raw QK^T softmax)
  qktv     sum_h α_h * ||W_O^h v_h||           (Kobayashi / value-weighted)

Then the T2IShield-style Frobenius dispersion on the 7 DoF rows:
  FTT = mean_i || p_i - mean_j p_j ||_2   (rows sum-normalized)

Also Gram Frobenius on L2-normalized rows: ||C - I||_F.

Plots + npz -> attn_maps/action_imgtext_qktv/
"""
import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from libero.libero import benchmark
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.libero.run_libero_eval_attentionmap import crop_and_resize
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

CHECKPOINT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
OUT = f"{REPO}/attn_maps/action_imgtext_qktv"
BDDL = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
SUITE = "libero_goal"
SEED = 7
NUM_STEPS_WAIT = 10
DEVICE = 0
N_DOF = 7
LAYER = -1  # last layer


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


def gram_fro(P):
    P = np.clip(P, 0, None)
    P = P / np.clip(P.sum(-1, keepdims=True), 1e-12, None)
    N = P / np.clip(np.linalg.norm(P, axis=1, keepdims=True), 1e-12, None)
    C = N @ N.T
    return float(np.linalg.norm(C - np.eye(len(P)), "fro"))


def get_scene(task, bddl, seed):
    env, desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False,
                               bddl_path=bddl, seed=seed)
    env.reset()
    obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
    img = preprocess(get_libero_image(obs, 224))
    env.close()
    return img, desc


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    os.makedirs(OUT, exist_ok=True)

    print("Loading GoBA model...", flush=True)
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        CHECKPOINT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE)
    vla.eval()
    n_patch = vla.vision_backbone.featurizer.patch_embed.num_patches

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
    print(f"patches={n_patch} layers={n_layers} heads={n_heads}", flush=True)

    task_suite = benchmark.get_benchmark_dict()[SUITE]()
    records = []

    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        for cond, bddl in BDDL.items():
            img, desc = get_scene(task, bddl, SEED)
            prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
            inputs = processor(prompt, Image.fromarray(img)).to(DEVICE, dtype=torch.bfloat16)
            input_ids = inputs.input_ids
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)
            n_lang = input_ids.shape[-1] - 1
            t0 = 1 + n_patch
            t1 = t0 + n_lang
            # key columns = image patches + text (contiguous after BOS)
            k_cols = list(range(1, t1))  # length n_patch + n_lang

            v_cache.clear()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                gen = vla.generate(
                    input_ids, pixel_values=inputs.pixel_values,
                    max_new_tokens=N_DOF, output_attentions=True,
                    return_dict_in_generate=True, do_sample=False,
                )

            # Build (7, K) weight and qktv maps from the last LLM layer at each action step.
            # For autoregressive generate, attentions[k][-1] is layer L at step k;
            # the query is the newly generated action token (last position).
            # Values: use v_cache from the LAST generate step's forward (covers full seq).
            # Safer: re-forward teacher-forced with the generated action tokens to get
            # aligned attentions + values for all 7 action rows at once.
            seq = gen.sequences
            del gen
            torch.cuda.empty_cache()

            # teacher-forced pass over prompt + 7 action tokens
            v_cache.clear()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = vla(input_ids=seq, pixel_values=inputs.pixel_values,
                          output_attentions=True, return_dict=True)

            N = out.attentions[0].shape[-1]
            # action token positions: last 7
            q_rows = list(range(N - N_DOF, N))
            assert len(k_cols) == n_patch + n_lang

            l = n_layers + LAYER if LAYER < 0 else LAYER
            A = out.attentions[l][0].float()                  # (H, N, N)
            vh = v_cache[l][0].float().view(N, n_heads, head_dim)
            f = torch.einsum("nhd,ohd->nho", vh, Wo[l].float())
            f_norms = f.norm(dim=-1)                         # (N, H)

            Aq = A[:, q_rows, :]                             # (H, 7, N)
            w = Aq.mean(0)[:, k_cols].cpu().numpy()          # (7, K) weight
            n = torch.einsum("hqn,nh->qn", Aq, f_norms)[:, k_cols].cpu().numpy()  # (7, K) qktv

            w = w / np.clip(w.sum(1, keepdims=True), 1e-12, None)
            n = n / np.clip(n.sum(1, keepdims=True), 1e-12, None)

            rec = dict(
                model="GoBA", task_id=task_id, seed=SEED, cond=cond,
                n_lang=n_lang, n_patch=n_patch,
                ftt_weight=ftt(w), ftt_qktv=ftt(n),
                gram_weight=gram_fro(w), gram_qktv=gram_fro(n),
                map_weight=w.astype(np.float32), map_qktv=n.astype(np.float32),
                rgb=img,
            )
            records.append(rec)
            print(f"GoBA task={task_id} {cond:6s} n_lang={n_lang}  "
                  f"FTT_w={rec['ftt_weight']:.4f} FTT_qktv={rec['ftt_qktv']:.4f}  "
                  f"Gram_w={rec['gram_weight']:.3f} Gram_qktv={rec['gram_qktv']:.3f}", flush=True)
            del out, A, vh, f, f_norms
            torch.cuda.empty_cache()

    for h in handles:
        h.remove()
    del vla
    torch.cuda.empty_cache()

    np.savez_compressed(
        f"{OUT}/goba.npz",
        task_id=np.array([r["task_id"] for r in records]),
        seed=np.array([r["seed"] for r in records]),
        cond=np.array([r["cond"] for r in records]),
        ftt_weight=np.array([r["ftt_weight"] for r in records]),
        ftt_qktv=np.array([r["ftt_qktv"] for r in records]),
        gram_weight=np.array([r["gram_weight"] for r in records]),
        gram_qktv=np.array([r["gram_qktv"] for r in records]),
        map_weight=np.stack([r["map_weight"] for r in records]),
        map_qktv=np.stack([r["map_qktv"] for r in records]),
        n_lang=np.array([r["n_lang"] for r in records]),
        n_patch=np.array([r["n_patch"] for r in records]),
    )
    print(f"Saved {len(records)} -> {OUT}/goba.npz", flush=True)


if __name__ == "__main__":
    main()
