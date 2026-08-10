"""
GoBA closed-loop: VALUE-WEIGHTED (α · ||W_O v||) text↔image and action↔image maps.

Same protocol / viz keys as weight-only rollouts, but each entry is
  sum_h  alpha_h[q,j] * ||W_O^h v_h(j)||
then row-normalized over image patches (Kobayashi / our norm-based pipeline).

Outputs:
  attn_maps/single_sample_analysis/goba_rollout_xattn_vnorm/
  attn_maps/single_sample_analysis/goba_rollout_action_xattn_vnorm/
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
from prismatic.vla.action_tokenizer import ActionTokenizer

GOBA_CKPT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
CLEAN_CKPT = "openvla/openvla-7b-finetuned-libero-goal"
OUT_TEXT = f"{REPO}/attn_maps/single_sample_analysis/goba_rollout_xattn_vnorm"
OUT_ACT = f"{REPO}/attn_maps/single_sample_analysis/goba_rollout_action_xattn_vnorm"
BDDL = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
SUITE = "libero_goal"
TASK_ID, SEED = 7, 7
NUM_STEPS_WAIT, NUM_STEPS, DEVICE, N_DOF = 10, 20, 0, 7
LAYER = -1
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


def attach_vhooks(vla):
    llm = vla.language_model
    layers = llm.model.layers
    attn0 = layers[0].self_attn
    n_heads, head_dim = attn0.num_heads, attn0.head_dim
    hidden = n_heads * head_dim
    assert getattr(attn0, "num_key_value_heads", n_heads) == n_heads
    v_cache = {}
    handles = []
    for l, lyr in enumerate(layers):
        handles.append(lyr.self_attn.v_proj.register_forward_hook(
            lambda _m, _i, out, l=l: v_cache.__setitem__(l, out)))
    Wo = {l: layers[l].self_attn.o_proj.weight.detach().view(hidden, n_heads, head_dim)
          for l in range(len(layers))}
    return v_cache, handles, Wo, n_heads, head_dim


def value_weighted_maps(A, vh, Wo_l, q_rows, k_cols):
    """A: (H,N,N), vh: (N,H,D) -> (len(q), len(k)) = sum_h α * ||Wo^h v_h||."""
    f = torch.einsum("nhd,ohd->nho", vh, Wo_l.float())
    f_norms = f.norm(dim=-1)  # (N, H)
    Aq = A[:, q_rows, :]      # (H, Q, N)
    m = torch.einsum("hqn,nh->qn", Aq, f_norms)[:, k_cols]
    return m


def row_norm(x):
    x = x.float().cpu().numpy().astype(np.float32)
    return x / np.clip(x.sum(1, keepdims=True), 1e-12, None)


def col_norm_from_rows(A, vh, Wo_l, q_rows, k_cols):
    """Patch→query: for each patch as query, mass on text/action keys (may be causal-weak)."""
    f = torch.einsum("nhd,ohd->nho", vh, Wo_l.float())
    f_norms = f.norm(dim=-1)
    Aq = A[:, k_cols, :]  # patches as queries
    m = torch.einsum("hqn,nh->qn", Aq, f_norms)[:, q_rows]  # (n_patch, n_q)
    m = m.float().cpu().numpy().astype(np.float32)
    return m / np.clip(m.sum(1, keepdims=True), 1e-12, None)


def run_condition(name, ckpt, bddl_key, task, shared_desc):
    print(f"\n=== {name} bddl={bddl_key} ===", flush=True)
    processor, vla = load_model(ckpt)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    unnorm_key = SUITE if SUITE in vla.norm_stats else f"{SUITE}_no_noops"
    stats = vla.get_action_stats(unnorm_key)
    a_low, a_high = np.array(stats["q01"]), np.array(stats["q99"])
    a_mask = np.array(stats.get("mask", np.ones_like(a_low, dtype=bool)))
    n_patch = vla.vision_backbone.featurizer.patch_embed.num_patches
    v_cache, handles, Wo, n_heads, head_dim = attach_vhooks(vla)
    li = LAYER if LAYER >= 0 else len(vla.language_model.model.layers) + LAYER

    env, _ = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False,
                            bddl_path=BDDL[bddl_key], seed=SEED)
    env.reset(); obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))

    t2ps, p2ts, a2ps, p2as, rgbs, acts = [], [], [], [], [], []
    toks = None; prompt = None; done = False
    for t in range(NUM_STEPS):
        img = preprocess(get_libero_image(obs, 224))
        prompt = f"In: What action should the robot take to {shared_desc.lower()}?\nOut:"
        inputs = processor(prompt, Image.fromarray(img)).to(DEVICE, dtype=torch.bfloat16)

        # --- TEXT: teacher-forced (no 29871), value-weighted ---
        v_cache.clear()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = vla(**inputs, output_attentions=True)
        A = out.attentions[li][0].float()
        N = A.shape[-1]
        vh = v_cache[li][0].float().view(N, n_heads, head_dim)
        mask = (inputs.input_ids < action_tokenizer.action_token_begin_idx)[0]
        # layout: [BOS][patches][text 1..]
        text_rows = (1 + n_patch + np.nonzero(mask.cpu().numpy()[1:])[0]).tolist()
        patch_cols = list(range(1, 1 + n_patch))
        t2p = row_norm(value_weighted_maps(A, vh, Wo[li], text_rows, patch_cols))
        p2t = col_norm_from_rows(A, vh, Wo[li], text_rows, patch_cols)
        ids = inputs.input_ids[0].tolist()
        kept = [tid for i, tid in enumerate(ids) if i > 0 and bool(mask[i])]
        toks = processor.tokenizer.convert_ids_to_tokens(kept)
        n = min(len(toks), t2p.shape[0])
        toks, t2p, p2t = toks[:n], t2p[:n], p2t[:, :n]
        del out

        # --- ACTION: generate then teacher-forced, value-weighted ---
        input_ids = inputs.input_ids
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = vla.generate(input_ids, pixel_values=inputs.pixel_values,
                               max_new_tokens=N_DOF, do_sample=False)
        action_ids = gen[0, -N_DOF:]
        full_ids = torch.cat((input_ids, action_ids.unsqueeze(0)), dim=1)
        v_cache.clear()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out2 = vla(input_ids=full_ids, pixel_values=inputs.pixel_values, output_attentions=True)
        A2 = out2.attentions[li][0].float()
        N2 = A2.shape[-1]
        vh2 = v_cache[li][0].float().view(N2, n_heads, head_dim)
        Lp = input_ids.shape[1]
        # rows that *produce* each action token = positions before that token
        q_rows = [n_patch + Lp - 1 + k for k in range(N_DOF)]
        assert N2 == n_patch + Lp + N_DOF, f"seq mismatch {N2}"
        a2p = row_norm(value_weighted_maps(A2, vh2, Wo[li], q_rows, patch_cols))
        p2a = np.zeros((n_patch, N_DOF), np.float32)  # placeholder (viz skips)

        tok = action_ids.cpu().numpy()
        disc = np.clip(vla.vocab_size - tok - 1, 0, vla.bin_centers.shape[0] - 1)
        normed = vla.bin_centers[disc]
        action = np.where(a_mask, 0.5 * (normed + 1) * (a_high - a_low) + a_low, normed)
        del gen, out2
        torch.cuda.empty_cache()

        t2ps.append(t2p); p2ts.append(p2t); a2ps.append(a2p); p2as.append(p2a)
        rgbs.append(img); acts.append(action.astype(np.float64))
        print(f"  step={t:02d} t2p max={t2p.max():.4f} a2p max={a2p.max():.4f} "
              f"a2p_fnorm={np.linalg.norm(a2p - a2p.mean(0), axis=1).mean():.4f}", flush=True)
        exec_a = invert_gripper_action(normalize_gripper_action(action.copy(), binarize=True))
        obs, _, done, _ = env.step(exec_a.tolist())
        if done:
            break
    env.close()
    for h in handles:
        h.remove()
    del vla, processor
    torch.cuda.empty_cache()
    return dict(
        text2patch=np.stack(t2ps), patch2text=np.stack(p2ts),
        action2patch=np.stack(a2ps), patch2action=np.stack(p2as),
        rgb=np.stack(rgbs), actions=np.stack(acts),
        tokens=np.array(toks, dtype=object),
        prompt=prompt, n_steps=len(t2ps), done=done,
    )


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    os.makedirs(OUT_TEXT, exist_ok=True)
    os.makedirs(OUT_ACT, exist_ok=True)
    task = benchmark.get_benchmark_dict()[SUITE]().get_task(TASK_ID)
    env_tmp, shared_desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False,
                                          bddl_path=BDDL["clean"], seed=SEED)
    env_tmp.close()
    print(f"shared_desc={shared_desc!r}", flush=True)

    clean = run_condition("CLEAN_model", CLEAN_CKPT, "clean", task, shared_desc)
    poison = run_condition("GOBA_backdoored", GOBA_CKPT, "poison", task, shared_desc)

    ntok = min(len(clean["tokens"]), len(poison["tokens"]),
               clean["text2patch"].shape[1], poison["text2patch"].shape[1])
    note = np.array("VALUE-WEIGHTED: sum_h alpha_h * ||W_O^h v_h|| ; row-normalized over patches")

    np.savez_compressed(
        f"{OUT_TEXT}/rollout_xattn.npz",
        clean_text2patch=clean["text2patch"][:, :ntok],
        clean_patch2text=clean["patch2text"][:, :, :ntok],
        poison_text2patch=poison["text2patch"][:, :ntok],
        poison_patch2text=poison["patch2text"][:, :, :ntok],
        clean_rgb=clean["rgb"], poison_rgb=poison["rgb"],
        clean_actions=clean["actions"], poison_actions=poison["actions"],
        tokens=clean["tokens"][:ntok],
        prompt=np.array(clean["prompt"]),
        task_id=TASK_ID, seed=SEED, layer=np.array([LAYER]),
        note=note,
    )
    np.savez_compressed(
        f"{OUT_ACT}/rollout_xattn.npz",
        clean_text2patch=clean["action2patch"],
        clean_patch2text=clean["patch2action"],
        poison_text2patch=poison["action2patch"],
        poison_patch2text=poison["patch2action"],
        clean_rgb=clean["rgb"], poison_rgb=poison["rgb"],
        clean_actions=clean["actions"], poison_actions=poison["actions"],
        tokens=np.array(DOF_NAMES, dtype=object),
        prompt=np.array(clean["prompt"]),
        task_id=TASK_ID, seed=SEED, layer=np.array([LAYER]),
        note=note,
    )
    print(f"\nSaved -> {OUT_TEXT}/rollout_xattn.npz", flush=True)
    print(f"Saved -> {OUT_ACT}/rollout_xattn.npz", flush=True)


if __name__ == "__main__":
    main()
