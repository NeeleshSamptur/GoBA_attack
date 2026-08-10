"""
Clean OpenVLA: first-pass action→image attention per DoF.
Shows raw weight and QKTV (α · ‖W_O v‖) overlays.
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
from scipy.ndimage import zoom
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.libero.run_libero_eval_attentionmap import crop_and_resize
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

CLEAN_CKPT = "openvla/openvla-7b-finetuned-libero-goal"
OUT = f"{REPO}/attn_maps/single_sample_analysis/clean_model_qktv_firstpass"
SUITE = "libero_goal"
TASK_ID = 7
SEED = 7
NUM_STEPS_WAIT = 10
DEVICE = 0
N_DOF = 7
LAYER = -1
GRID = 16
DOF_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def preprocess(img):
    im = tf.convert_to_tensor(np.array(Image.fromarray(img).convert("RGB")))
    dt = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, dt, saturate=True)
    return np.array(Image.fromarray(im.numpy()).convert("RGB"))


def overlay_row(ax_row, img, maps, titles, vmin, vmax):
    for ax, m, title in zip(ax_row, maps, titles):
        up = zoom(m.reshape(GRID, GRID), img.shape[0] / GRID, order=1)
        ax.imshow(img)
        im = ax.imshow(up, cmap="jet", alpha=0.55, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    return im


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    os.makedirs(OUT, exist_ok=True)

    task = benchmark.get_benchmark_dict()[SUITE]().get_task(TASK_ID)
    bddl = f"{REPO}/BadLIBERO/libero/libero/bddl_files"
    env, desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False,
                               bddl_path=bddl, seed=SEED)
    env.reset()
    obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
    img = preprocess(get_libero_image(obs, 224))
    env.close()
    Image.fromarray(img).save(f"{OUT}/rgb_t{TASK_ID}_s{SEED}.png")
    print(f"desc={desc}", flush=True)

    print("Loading clean model...", flush=True)
    processor = AutoProcessor.from_pretrained(CLEAN_CKPT, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        CLEAN_CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE)
    vla.eval()
    n_patch = vla.vision_backbone.featurizer.patch_embed.num_patches

    llm = vla.language_model
    layers = llm.model.layers
    n_layers = len(layers)
    n_heads, head_dim = layers[0].self_attn.num_heads, layers[0].self_attn.head_dim
    hidden = n_heads * head_dim
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

    # First pass: generate 7 action tokens, then teacher-force full seq for attentions + V
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        gen = vla.generate(
            input_ids, pixel_values=inputs.pixel_values,
            max_new_tokens=N_DOF, output_attentions=False,
            return_dict_in_generate=True, do_sample=False,
        )
    seq = gen.sequences
    del gen
    v_cache.clear()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(input_ids=seq, pixel_values=inputs.pixel_values,
                  output_attentions=True, return_dict=True)

    N = out.attentions[0].shape[-1]
    q_rows = list(range(N - N_DOF, N))
    k_cols = list(range(1, 1 + n_patch))  # image patches only
    l = n_layers + LAYER if LAYER < 0 else LAYER
    A = out.attentions[l][0].float()
    vh = v_cache[l][0].float().view(N, n_heads, head_dim)
    f = torch.einsum("nhd,ohd->nho", vh, Wo[l].float())
    f_norms = f.norm(dim=-1)
    Aq = A[:, q_rows, :]
    w = Aq.mean(0)[:, k_cols].cpu().numpy()  # (7, 256)
    qktv = torch.einsum("hqn,nh->qn", Aq, f_norms)[:, k_cols].cpu().numpy()
    w = w / np.clip(w.sum(1, keepdims=True), 1e-12, None)
    qktv = qktv / np.clip(qktv.sum(1, keepdims=True), 1e-12, None)

    for h in handles:
        h.remove()
    del out
    torch.cuda.empty_cache()

    # --- figure: weight row + QKTV row, 7 DoFs + pooled ---
    maps_w = [w[k] for k in range(N_DOF)] + [w.mean(0)]
    maps_q = [qktv[k] for k in range(N_DOF)] + [qktv.mean(0)]
    titles_w = [f"{n}\nmax={m.max():.3f}" for n, m in zip(DOF_NAMES + ["POOLED"], maps_w)]
    titles_q = [f"{n}\nmax={m.max():.3f}" for n, m in zip(DOF_NAMES + ["POOLED"], maps_q)]

    fig, axes = plt.subplots(2, 8, figsize=(22, 6.2))
    all_w = np.concatenate([m.ravel() for m in maps_w])
    all_q = np.concatenate([m.ravel() for m in maps_q])
    vw = np.percentile(all_w, [1, 99]); vq = np.percentile(all_q, [1, 99])
    im0 = overlay_row(axes[0], img, maps_w, titles_w, vw[0], vw[1])
    im1 = overlay_row(axes[1], img, maps_q, titles_q, vq[0], vq[1])
    axes[0, 0].set_ylabel("weight α", fontsize=11)
    axes[1, 0].set_ylabel("QKTV α·‖Wₒv‖", fontsize=11)
    fig.colorbar(im0, ax=axes[0].tolist(), shrink=0.85, label="mass")
    fig.colorbar(im1, ax=axes[1].tolist(), shrink=0.85, label="mass")
    fig.suptitle(
        f"Clean OpenVLA · first pass · action→image · L{l} · task={TASK_ID} seed={SEED}\n{prompt.strip()}",
        fontsize=12,
    )
    fig.tight_layout()
    out_png = f"{OUT}/clean_model_action_img_weight_vs_qktv_per_dof.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # QKTV-only larger panel (what user asked for)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2))
    axes = axes.ravel()
    allv = np.concatenate([m.ravel() for m in maps_q])
    vmin, vmax = np.percentile(allv, [1, 99])
    for i, (m, title) in enumerate(zip(maps_q, titles_q)):
        up = zoom(m.reshape(GRID, GRID), img.shape[0] / GRID, order=1)
        axes[i].imshow(img)
        im = axes[i].imshow(up, cmap="jet", alpha=0.55, vmin=vmin, vmax=vmax)
        axes[i].set_title(f"QKTV · {title}", fontsize=10)
        axes[i].axis("off")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.7, label="QKTV mass")
    fig.suptitle(
        f"Clean model · QKTV action→image per DoF (first pass)\n{prompt.strip()}",
        fontsize=12,
    )
    fig.tight_layout()
    out_q = f"{OUT}/clean_model_action_img_qktv_per_dof.png"
    fig.savefig(out_q, dpi=160, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        f"{OUT}/maps.npz",
        weight=w.astype(np.float32), qktv=qktv.astype(np.float32),
        rgb=img, task_id=TASK_ID, seed=SEED, layer=l, prompt=np.array(prompt),
        dof_names=np.array(DOF_NAMES),
    )
    print(f"Saved {out_q}", flush=True)
    print(f"Saved {out_png}", flush=True)
    print(f"argmax QKTV per DoF: {[int(qktv[k].argmax()) for k in range(N_DOF)]}", flush=True)


if __name__ == "__main__":
    main()
