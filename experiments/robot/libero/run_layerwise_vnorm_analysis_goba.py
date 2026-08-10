"""
Layer-wise HEAD-AVERAGED value-weighted (α · ||W_O v||) analysis for GoBA.

Same scene as single-sample (task 7, seed 7). For every LLM layer:
  text→image and action→image maps (head-sum of α_h * ||Wo^h v_h||, row-normed)
  then entropy / f_norm / peak for CLEAN vs GOBA+trigger.

Output: attn_maps/single_sample_analysis/goba_layerwise_vnorm/
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
OUT = f"{REPO}/attn_maps/single_sample_analysis/goba_layerwise_vnorm"
BDDL = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
SUITE, TASK_ID, SEED = "libero_goal", 7, 7
NUM_STEPS_WAIT, DEVICE, N_DOF, GRID = 10, 0, 7, 16
DOF_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
# layers to dump full overlays (all layers get scalar stats)
SHOW_LAYERS = [0, 4, 8, 12, 16, 20, 24, 27, 31]


def preprocess(img):
    im = tf.convert_to_tensor(np.array(Image.fromarray(img).convert("RGB")))
    dt = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, dt, saturate=True)
    return np.array(Image.fromarray(im.numpy()).convert("RGB"))


def entropy_rows(M):
    p = M / np.clip(M.sum(-1, keepdims=True), 1e-12, None)
    return float((-(p * np.log(p + 1e-12)).sum(-1)).mean())


def fnorm_rows(M):
    p = M / np.clip(M.sum(-1, keepdims=True), 1e-12, None)
    return float(np.linalg.norm(p - p.mean(0), axis=1).mean())


def head_avg_vw(A, vh, Wo_l, q_rows, k_cols):
    """(Q,K) = sum_h α_h * ||Wo^h v_h||, then row-normalize."""
    f = torch.einsum("nhd,ohd->nho", vh, Wo_l.float())
    f_norms = f.norm(dim=-1)  # (N,H)
    Aq = A[:, q_rows, :]      # (H,Q,N)
    m = torch.einsum("hqn,nh->qn", Aq, f_norms)[:, k_cols]
    m = m.float().cpu().numpy().astype(np.float32)
    return m / np.clip(m.sum(1, keepdims=True), 1e-12, None)


def head_avg_w(A, q_rows, k_cols):
    m = A[:, q_rows, :][:, :, k_cols].mean(0).float().cpu().numpy().astype(np.float32)
    return m / np.clip(m.sum(1, keepdims=True), 1e-12, None)


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

    # TEXT forward
    v_cache.clear()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(**inputs, output_attentions=True)
    mask = (inputs.input_ids < action_tokenizer.action_token_begin_idx)[0]
    text_rows = (1 + n_patch + np.nonzero(mask.cpu().numpy()[1:])[0]).tolist()
    patch_cols = list(range(1, 1 + n_patch))
    ids = inputs.input_ids[0].tolist()
    toks = processor.tokenizer.convert_ids_to_tokens(
        [tid for i, tid in enumerate(ids) if i > 0 and bool(mask[i])])

    text_vw = np.zeros((n_layers, len(text_rows), n_patch), np.float32)
    text_w = np.zeros_like(text_vw)
    for l in range(n_layers):
        A = out.attentions[l][0].float()
        N = A.shape[-1]
        vh = v_cache[l][0].float().view(N, n_heads, head_dim)
        text_vw[l] = head_avg_vw(A, vh, Wo[l], text_rows, patch_cols)
        text_w[l] = head_avg_w(A, text_rows, patch_cols)
    del out

    # ACTION: generate + teacher force
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
        out2 = vla(input_ids=full_ids, pixel_values=inputs.pixel_values, output_attentions=True)
    Lp = input_ids.shape[1]
    q_rows = [n_patch + Lp - 1 + k for k in range(N_DOF)]
    act_vw = np.zeros((n_layers, N_DOF, n_patch), np.float32)
    act_w = np.zeros_like(act_vw)
    for l in range(n_layers):
        A = out2.attentions[l][0].float()
        N = A.shape[-1]
        vh = v_cache[l][0].float().view(N, n_heads, head_dim)
        act_vw[l] = head_avg_vw(A, vh, Wo[l], q_rows, patch_cols)
        act_w[l] = head_avg_w(A, q_rows, patch_cols)
    del gen, out2
    for h in handles:
        h.remove()
    del vla, processor
    torch.cuda.empty_cache()

    def pack(maps):
        return dict(
            entropy=np.array([entropy_rows(maps[l]) for l in range(n_layers)], np.float32),
            fnorm=np.array([fnorm_rows(maps[l]) for l in range(n_layers)], np.float32),
            peak=np.array([maps[l].max() for l in range(n_layers)], np.float32),
        )

    return dict(
        text_vw=text_vw, text_w=text_w, act_vw=act_vw, act_w=act_w,
        text_stats_vw=pack(text_vw), text_stats_w=pack(text_w),
        act_stats_vw=pack(act_vw), act_stats_w=pack(act_w),
        tokens=toks, prompt=prompt, rgb=img, n_layers=n_layers,
    )


def plot_layer_curves(c, p, out):
    L = c["n_layers"]
    x = np.arange(L)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for row, (key, title) in enumerate([
        ("text_stats_vw", "TEXT→img (α·‖Wov‖)"),
        ("act_stats_vw", "ACTION→img (α·‖Wov‖)"),
    ]):
        for col, stat, ylabel in [
            (0, "entropy", "entropy"),
            (1, "fnorm", "f_norm"),
            (2, "peak", "max mass"),
        ]:
            ax = axes[row, col]
            ax.plot(x, c[key][stat], "o-", label="clean", color="steelblue", ms=3)
            ax.plot(x, p[key][stat], "s-", label="bd+trig", color="darkorange", ms=3)
            ax.set_title(f"{title}: {ylabel}")
            ax.set_xlabel("layer"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("GoBA layer-wise head-averaged value-weighted attention")
    fig.tight_layout()
    fig.savefig(f"{out}/LAYER_curves_vnorm.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Δ curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in [
        (axes[0], "text_stats_vw", "TEXT→img"),
        (axes[1], "act_stats_vw", "ACTION→img"),
    ]:
        dent = p[key]["entropy"] - c[key]["entropy"]
        dfn = p[key]["fnorm"] - c[key]["fnorm"]
        ax.plot(x, dent, "o-", label="Δentropy (bd−clean)", color="crimson", ms=3)
        ax.plot(x, dfn, "s-", label="Δf_norm", color="purple", ms=3)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{title} layer deltas"); ax.set_xlabel("layer")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("GoBA: where layers diverge under trigger (head-avg α·‖Wov‖)")
    fig.tight_layout()
    fig.savefig(f"{out}/LAYER_deltas_vnorm.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # weight vs vnorm entropy overlay
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, sk_w, sk_v, title in [
        (axes[0], "text_stats_w", "text_stats_vw", "TEXT→img"),
        (axes[1], "act_stats_w", "act_stats_vw", "ACTION→img"),
    ]:
        ax.plot(x, c[sk_w]["entropy"], "o--", color="steelblue", alpha=0.5, label="clean weight", ms=3)
        ax.plot(x, p[sk_w]["entropy"], "s--", color="darkorange", alpha=0.5, label="bd weight", ms=3)
        ax.plot(x, c[sk_v]["entropy"], "o-", color="steelblue", label="clean α·‖v‖", ms=3)
        ax.plot(x, p[sk_v]["entropy"], "s-", color="darkorange", label="bd α·‖v‖", ms=3)
        ax.set_title(f"{title} entropy: weight vs value-weighted")
        ax.set_xlabel("layer"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/LAYER_weight_vs_vnorm_entropy.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved layer curves")


def plot_layer_overlays(c, p, out):
    layers = [l for l in SHOW_LAYERS if l < c["n_layers"]]
    for kind, key in [("text", "text_vw"), ("action", "act_vw")]:
        fig, axes = plt.subplots(2, len(layers), figsize=(2.4 * len(layers), 5.2))
        for j, l in enumerate(layers):
            Mc = c[key][l].mean(0)
            Mp = p[key][l].mean(0)
            for row, (M, rgb, tag) in enumerate([
                (Mc, c["rgb"], "CLEAN"), (Mp, p["rgb"], "BD+TRIG"),
            ]):
                ax = axes[row, j]
                up = zoom(M.reshape(GRID, GRID), rgb.shape[0] / GRID, order=1)
                ax.imshow(rgb); ax.imshow(up, cmap="jet", alpha=0.55)
                ax.set_title(f"L{l} {tag}\nmax={M.max():.3f}", fontsize=8)
                ax.axis("off")
        fig.suptitle(f"GoBA {kind.upper()}→img head-avg α·‖Wov‖ across layers")
        fig.tight_layout()
        fig.savefig(f"{out}/OVERLAY_{kind}_selected_layers.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved OVERLAY_{kind}_selected_layers.png")


def write_table(c, p, out):
    lines = ["layer  text_Δent  text_Δfn  act_Δent  act_Δfn  text_ent_c  text_ent_p  act_ent_c  act_ent_p"]
    for l in range(c["n_layers"]):
        te_c, te_p = c["text_stats_vw"]["entropy"][l], p["text_stats_vw"]["entropy"][l]
        ae_c, ae_p = c["act_stats_vw"]["entropy"][l], p["act_stats_vw"]["entropy"][l]
        tf_c, tf_p = c["text_stats_vw"]["fnorm"][l], p["text_stats_vw"]["fnorm"][l]
        af_c, af_p = c["act_stats_vw"]["fnorm"][l], p["act_stats_vw"]["fnorm"][l]
        lines.append(
            f"{l:5d}  {te_p-te_c:+8.3f}  {tf_p-tf_c:+7.4f}  {ae_p-ae_c:+8.3f}  {af_p-af_c:+7.4f}  "
            f"{te_c:9.3f}  {te_p:9.3f}  {ae_c:8.3f}  {ae_p:8.3f}"
        )
    # highlight top layers by |Δent|
    te = p["text_stats_vw"]["entropy"] - c["text_stats_vw"]["entropy"]
    ae = p["act_stats_vw"]["entropy"] - c["act_stats_vw"]["entropy"]
    lines.append("\nTop |Δentropy| TEXT layers: " +
                 ", ".join(f"L{i}({te[i]:+.3f})" for i in np.argsort(-np.abs(te))[:5]))
    lines.append("Top |Δentropy| ACTION layers: " +
                 ", ".join(f"L{i}({ae[i]:+.3f})" for i in np.argsort(-np.abs(ae))[:5]))
    with open(f"{out}/LAYER_table.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-3:]))


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
        f"{OUT}/layerwise.npz",
        clean_text_vw=clean["text_vw"], poison_text_vw=poison["text_vw"],
        clean_act_vw=clean["act_vw"], poison_act_vw=poison["act_vw"],
        clean_text_w=clean["text_w"], poison_text_w=poison["text_w"],
        clean_act_w=clean["act_w"], poison_act_w=poison["act_w"],
        clean_text_ent_vw=clean["text_stats_vw"]["entropy"],
        poison_text_ent_vw=poison["text_stats_vw"]["entropy"],
        clean_act_ent_vw=clean["act_stats_vw"]["entropy"],
        poison_act_ent_vw=poison["act_stats_vw"]["entropy"],
        clean_text_fn_vw=clean["text_stats_vw"]["fnorm"],
        poison_text_fn_vw=poison["text_stats_vw"]["fnorm"],
        clean_act_fn_vw=clean["act_stats_vw"]["fnorm"],
        poison_act_fn_vw=poison["act_stats_vw"]["fnorm"],
        clean_rgb=clean["rgb"], poison_rgb=poison["rgb"],
        tokens=np.array(clean["tokens"], dtype=object),
        prompt=np.array(clean["prompt"]),
    )
    plot_layer_curves(clean, poison, OUT)
    plot_layer_overlays(clean, poison, OUT)
    write_table(clean, poison, OUT)
    with open(f"{OUT}/README.txt", "w") as f:
        f.write("Head-averaged value-weighted attention across all LLM layers.\n")
        f.write(f"n_layers={clean['n_layers']}  prompt={clean['prompt']}\n")
    print(f"\nDone -> {OUT}")


if __name__ == "__main__":
    main()
