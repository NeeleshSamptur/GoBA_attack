"""
Attention ROLLOUT (Abnar & Zuidema 2020) for GoBA — multi-layer flow.

Â^(l) = 0.5 A^(l) + 0.5 I
R = Â^(L-1) @ ... @ Â^(0)

Head-averaged WEIGHT and VALUE-WEIGHTED (sum_h α·||W_O v||) rollouts for
text→image and action→image, plus last-layer snapshots for comparison.

Output: attn_maps/single_sample_analysis/goba_attention_rollout/
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from libero.libero import benchmark
from PIL import Image
from scipy.ndimage import zoom
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.libero.run_libero_eval_attentionmap import crop_and_resize
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer

GOBA_CKPT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
CLEAN_CKPT = "openvla/openvla-7b-finetuned-libero-goal"
OUT = f"{REPO}/attn_maps/single_sample_analysis/goba_attention_rollout"
BDDL = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
SUITE, TASK_ID, SEED = "libero_goal", 7, 7
NUM_STEPS_WAIT, DEVICE, N_DOF, GRID = 10, 0, 7, 16
DOF_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def preprocess(img):
    im = tf.convert_to_tensor(np.array(Image.fromarray(img).convert("RGB")))
    dt = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, dt, saturate=True)
    return np.array(Image.fromarray(im.numpy()).convert("RGB"))


def lab(t):
    s = str(t).replace("▁", " ").replace("Ġ", " ").replace("<0x0A>", "\\n").strip()
    return s or "∅"


def entropy_rows(M):
    p = M / np.clip(M.sum(-1, keepdims=True), 1e-12, None)
    return float((-(p * np.log(p + 1e-12)).sum(-1)).mean())


def fnorm_rows(M):
    p = M / np.clip(M.sum(-1, keepdims=True), 1e-12, None)
    return float(np.linalg.norm(p - p.mean(0), axis=1).mean())


def row_norm(M):
    return M / np.clip(M.sum(-1, keepdims=True), 1e-12, None)


def get_scene(bddl_key):
    task = benchmark.get_benchmark_dict()[SUITE]().get_task(TASK_ID)
    env, desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False,
                               bddl_path=BDDL[bddl_key], seed=SEED)
    env.reset(); obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
    img = preprocess(get_libero_image(obs, 224))
    env.close()
    return img, desc


def rollout_from_attentions(attentions, v_cache, Wo, n_heads, head_dim, device):
    n_layers = len(attentions)
    N = attentions[0].shape[-1]
    eye = torch.eye(N, device=device)
    R_w = eye.clone()
    R_v = eye.clone()
    last_w = last_v = None
    for l in range(n_layers):
        A_h = attentions[l][0].float()
        vh = v_cache[l][0].float().view(N, n_heads, head_dim)
        f = torch.einsum("nhd,ohd->nho", vh, Wo[l].float())
        f_norms = f.norm(dim=-1)
        A_w = A_h.mean(0)
        A_v = torch.einsum("hij,jh->ij", A_h, f_norms)
        A_v = A_v / A_v.sum(-1, keepdim=True).clamp_min(1e-12)
        last_w, last_v = A_w, A_v
        R_w = (0.5 * A_w + 0.5 * eye) @ R_w
        R_v = (0.5 * A_v + 0.5 * eye) @ R_v
    return R_w, R_v, last_w, last_v


def probe(ckpt, img, desc):
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE)
    if os.path.isdir(ckpt) and os.path.exists(os.path.join(ckpt, "dataset_statistics.json")):
        vla.norm_stats = json.load(open(os.path.join(ckpt, "dataset_statistics.json")))
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    n_patch = vla.vision_backbone.featurizer.patch_embed.num_patches
    layers = vla.language_model.model.layers
    n_layers = len(layers)
    attn0 = layers[0].self_attn
    n_heads, head_dim = attn0.num_heads, attn0.head_dim
    hidden = n_heads * head_dim
    assert getattr(attn0, "num_key_value_heads", n_heads) == n_heads
    v_cache = {}
    handles = [lyr.self_attn.v_proj.register_forward_hook(
        lambda _m, _i, out, l=l: v_cache.__setitem__(l, out))
        for l, lyr in enumerate(layers)]
    Wo = {l: layers[l].self_attn.o_proj.weight.detach().view(hidden, n_heads, head_dim)
          for l in range(n_layers)}

    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, Image.fromarray(img)).to(DEVICE, dtype=torch.bfloat16)
    input_ids = inputs.input_ids
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        gen = vla.generate(input_ids, pixel_values=inputs.pixel_values,
                           max_new_tokens=N_DOF, do_sample=False)
    full_ids = torch.cat((input_ids, gen[0, -N_DOF:].unsqueeze(0)), dim=1)
    v_cache.clear()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(input_ids=full_ids, pixel_values=inputs.pixel_values, output_attentions=True)

    R_w, R_v, last_w, last_v = rollout_from_attentions(
        out.attentions, v_cache, Wo, n_heads, head_dim, DEVICE)

    Lp = input_ids.shape[1]
    mask = (input_ids < action_tokenizer.action_token_begin_idx)[0]
    text_rows = (1 + n_patch + np.nonzero(mask.cpu().numpy()[1:])[0]).tolist()
    act_rows = [n_patch + Lp - 1 + k for k in range(N_DOF)]
    img_sl = slice(1, 1 + n_patch)

    def slice_maps(R, last):
        return (
            row_norm(R[text_rows, img_sl].float().cpu().numpy()),
            row_norm(R[act_rows, img_sl].float().cpu().numpy()),
            row_norm(last[text_rows, img_sl].float().cpu().numpy()),
            row_norm(last[act_rows, img_sl].float().cpu().numpy()),
        )

    tw_roll, aw_roll, tw_last, aw_last = slice_maps(R_w, last_w)
    tv_roll, av_roll, tv_last, av_last = slice_maps(R_v, last_v)

    ids = input_ids[0].tolist()
    toks = [lab(t) for i, t in enumerate(
        processor.tokenizer.convert_ids_to_tokens(ids)) if i > 0 and bool(mask[i])]
    n = min(len(toks), tw_roll.shape[0])
    toks = toks[:n]
    tw_roll, tw_last = tw_roll[:n], tw_last[:n]
    tv_roll, tv_last = tv_roll[:n], tv_last[:n]

    del gen, out
    for h in handles:
        h.remove()
    del vla, processor
    torch.cuda.empty_cache()
    return dict(
        text_roll_w=tw_roll, act_roll_w=aw_roll, text_last_w=tw_last, act_last_w=aw_last,
        text_roll_v=tv_roll, act_roll_v=av_roll, text_last_v=tv_last, act_last_v=av_last,
        tokens=toks, prompt=prompt, rgb=img, n_layers=n_layers,
    )


def plot_compare(c, p, out):
    for mode, roll_k, last_k, title in [
        ("text_v", "text_roll_v", "text_last_v", "TEXT→img α·‖Wov‖"),
        ("act_v", "act_roll_v", "act_last_v", "ACTION→img α·‖Wov‖"),
        ("text_w", "text_roll_w", "text_last_w", "TEXT→img weight"),
        ("act_w", "act_roll_w", "act_last_w", "ACTION→img weight"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        for col, (d, tag) in enumerate([(c, "CLEAN"), (p, "BD+TRIG")]):
            for row, (key, kind) in enumerate([(last_k, "LAST layer"), (roll_k, "ROLLOUT")]):
                M = d[key].mean(0)
                rgb = d["rgb"]
                ax = axes[row, col]
                up = zoom(M.reshape(GRID, GRID), rgb.shape[0] / GRID, order=1)
                ax.imshow(rgb); ax.imshow(up, cmap="jet", alpha=0.55)
                ax.set_title(f"{tag} | {kind}\nmax={M.max():.3f} H={entropy_rows(d[key]):.2f}", fontsize=9)
                ax.axis("off")
        fig.suptitle(f"GoBA {title}: last-layer vs attention rollout")
        fig.tight_layout()
        fig.savefig(f"{out}/COMPARE_{mode}_last_vs_rollout.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    vmin = min(np.percentile(c["act_last_v"], 1), np.percentile(p["act_last_v"], 1),
               np.percentile(c["act_roll_v"], 1), np.percentile(p["act_roll_v"], 1))
    vmax = max(np.percentile(c["act_last_v"], 99), np.percentile(p["act_last_v"], 99),
               np.percentile(c["act_roll_v"], 99), np.percentile(p["act_roll_v"], 99))
    for col, (d, tag) in enumerate([(c, "CLEAN"), (p, "BD+TRIG")]):
        for row, (key, kind) in enumerate([("act_last_v", "LAST"), ("act_roll_v", "ROLLOUT")]):
            M = d[key]
            ax = axes[row, col]
            im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_yticks(range(N_DOF)); ax.set_yticklabels(DOF_NAMES, fontsize=8)
            ax.set_title(f"{tag} ACTION {kind}  H={entropy_rows(M):.2f} fn={fnorm_rows(M):.3f}")
            ax.set_xlabel("patch")
        fig.colorbar(im, ax=axes[row].tolist(), shrink=0.8)
    fig.suptitle("GoBA ACTION→img value-weighted: last vs rollout")
    fig.tight_layout()
    fig.savefig(f"{out}/MATRIX_action_v_last_vs_rollout.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    labels = ["text W last", "text W roll", "text V last", "text V roll",
              "act W last", "act W roll", "act V last", "act V roll"]
    keys = ["text_last_w", "text_roll_w", "text_last_v", "text_roll_v",
            "act_last_w", "act_roll_w", "act_last_v", "act_roll_v"]
    ent_c = [entropy_rows(c[k]) for k in keys]
    ent_p = [entropy_rows(p[k]) for k in keys]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - 0.2, ent_c, 0.4, label="clean", color="steelblue")
    ax.bar(x + 0.2, ent_p, 0.4, label="bd+trig", color="darkorange")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("mean row entropy"); ax.legend()
    ax.set_title("GoBA: last-layer vs ROLLOUT entropy (higher = flatter)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/BARS_entropy_last_vs_rollout.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    lines = ["GoBA attention rollout vs last layer\n"]
    lines.append(f"{'map':20s} {'H_clean':>8} {'H_bd':>8} {'ΔH':>8} {'fn_c':>8} {'fn_p':>8}")
    for lab_, k in zip(labels, keys):
        hc, hp = entropy_rows(c[k]), entropy_rows(p[k])
        fc, fp = fnorm_rows(c[k]), fnorm_rows(p[k])
        lines.append(f"{lab_:20s} {hc:8.3f} {hp:8.3f} {hp-hc:+8.3f} {fc:8.4f} {fp:8.4f}")
    with open(f"{out}/SUMMARY.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    os.makedirs(OUT, exist_ok=True)

    img_c, desc = get_scene("clean")
    img_p, _ = get_scene("poison")
    print(f"desc={desc!r}", flush=True)
    print("\n=== CLEAN ===", flush=True)
    clean = probe(CLEAN_CKPT, img_c, desc)
    print("\n=== GOBA ===", flush=True)
    poison = probe(GOBA_CKPT, img_p, desc)

    np.savez_compressed(
        f"{OUT}/rollout.npz",
        clean_text_roll_w=clean["text_roll_w"], poison_text_roll_w=poison["text_roll_w"],
        clean_act_roll_w=clean["act_roll_w"], poison_act_roll_w=poison["act_roll_w"],
        clean_text_roll_v=clean["text_roll_v"], poison_text_roll_v=poison["text_roll_v"],
        clean_act_roll_v=clean["act_roll_v"], poison_act_roll_v=poison["act_roll_v"],
        clean_text_last_w=clean["text_last_w"], poison_text_last_w=poison["text_last_w"],
        clean_act_last_w=clean["act_last_w"], poison_act_last_w=poison["act_last_w"],
        clean_text_last_v=clean["text_last_v"], poison_text_last_v=poison["text_last_v"],
        clean_act_last_v=clean["act_last_v"], poison_act_last_v=poison["act_last_v"],
        clean_rgb=clean["rgb"], poison_rgb=poison["rgb"],
        tokens=np.array(clean["tokens"], dtype=object),
        dof_names=np.array(DOF_NAMES, dtype=object),
        prompt=np.array(clean["prompt"]),
        n_layers=clean["n_layers"],
        note=np.array("Abnar rollout R=(0.5A+0.5I)@... ; V uses sum_h α||Wov||"),
    )
    plot_compare(clean, poison, OUT)
    with open(f"{OUT}/README.txt", "w") as f:
        f.write("Attention ROLLOUT (Abnar & Zuidema 2020) vs last-layer snapshot.\n")
        f.write("Â=0.5A+0.5I per layer; R = Â_{L-1}@...@Â_0.\n")
        f.write("Weight = head-avg α; Value-weighted = sum_h α_h·||W_O^h v_h|| then row-norm.\n")
        f.write(f"n_layers={clean['n_layers']}  prompt={clean['prompt']}\n")
    print(f"\nDone -> {OUT}")


if __name__ == "__main__":
    main()
