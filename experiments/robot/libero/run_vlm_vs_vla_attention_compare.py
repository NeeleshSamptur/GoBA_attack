"""
Same scene → clean VLM / clean VLA / backdoored VLA attention stats.

Uses the t=0 RGB from goba_rollout_xattn_vnorm (and poison scene).
Models (same OpenVLA architecture family for fair comparison):
  1) openvla/openvla-7b                     -- pretrained base (Bridge) = "clean VLM/VLA"
  2) openvla/openvla-7b-finetuned-libero-goal -- clean LIBERO policy
  3) GoBA backdoored checkpoint

For each: last-layer head-avg WEIGHT and value-weighted (α·||Wov||) text→image
maps on the shared prompt; plus action→image after generate (if possible).

Output: attn_maps/single_sample_analysis/vlm_vs_vla_compare/
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
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer

GOBA_CKPT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
CLEAN_VLA = "openvla/openvla-7b-finetuned-libero-goal"
BASE_VLM = "openvla/openvla-7b"  # pretrained, not LIBERO-finetuned / not backdoored
SCENE_NPZ = f"{REPO}/attn_maps/single_sample_analysis/goba_rollout_xattn_vnorm/rollout_xattn.npz"
OUT = f"{REPO}/attn_maps/single_sample_analysis/vlm_vs_vla_compare"
DEVICE, N_DOF, GRID, LAYER = 0, 7, 16, -1


def entropy_rows(M):
    p = M / np.clip(M.sum(-1, keepdims=True), 1e-12, None)
    return float((-(p * np.log(p + 1e-12)).sum(-1)).mean())


def fnorm_rows(M):
    p = M / np.clip(M.sum(-1, keepdims=True), 1e-12, None)
    return float(np.linalg.norm(p - p.mean(0), axis=1).mean())


def peak(M):
    return float(M.max())


def row_norm(M):
    return M / np.clip(M.sum(-1, keepdims=True), 1e-12, None)


def load_model(ckpt):
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE)
    stats_path = os.path.join(ckpt, "dataset_statistics.json") if os.path.isdir(ckpt) else None
    if stats_path and os.path.exists(stats_path):
        vla.norm_stats = json.load(open(stats_path))
    return processor, vla


def probe(ckpt, img, prompt_desc, name):
    print(f"\n=== {name} ===", flush=True)
    processor, vla = load_model(ckpt)
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
    li = LAYER if LAYER >= 0 else n_layers + LAYER

    prompt = f"In: What action should the robot take to {prompt_desc.lower()}?\nOut:"
    inputs = processor(prompt, Image.fromarray(img)).to(DEVICE, dtype=torch.bfloat16)

    # TEXT → image (teacher-forced, no 29871)
    v_cache.clear()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(**inputs, output_attentions=True)
    A = out.attentions[li][0].float()
    N = A.shape[-1]
    vh = v_cache[li][0].float().view(N, n_heads, head_dim)
    f = torch.einsum("nhd,ohd->nho", vh, Wo[li].float())
    f_norms = f.norm(dim=-1)
    mask = (inputs.input_ids < action_tokenizer.action_token_begin_idx)[0]
    text_rows = (1 + n_patch + np.nonzero(mask.cpu().numpy()[1:])[0]).tolist()
    patch_cols = list(range(1, 1 + n_patch))
    Aq = A[:, text_rows, :]
    t_w = row_norm(Aq.mean(0)[:, patch_cols].cpu().numpy())
    t_v = row_norm(torch.einsum("hqn,nh->qn", Aq, f_norms)[:, patch_cols].cpu().numpy())
    del out

    # ACTION → image (generate + teacher force) — may fail / be meaningless for base if bins differ
    act_ok = True
    try:
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
        A2 = out2.attentions[li][0].float()
        N2 = A2.shape[-1]
        vh2 = v_cache[li][0].float().view(N2, n_heads, head_dim)
        f2 = torch.einsum("nhd,ohd->nho", vh2, Wo[li].float())
        fn2 = f2.norm(dim=-1)
        Lp = input_ids.shape[1]
        q_rows = [n_patch + Lp - 1 + k for k in range(N_DOF)]
        Aq2 = A2[:, q_rows, :]
        a_w = row_norm(Aq2.mean(0)[:, patch_cols].cpu().numpy())
        a_v = row_norm(torch.einsum("hqn,nh->qn", Aq2, fn2)[:, patch_cols].cpu().numpy())
        del gen, out2
    except Exception as e:
        print(f"  action path failed: {e}", flush=True)
        act_ok = False
        a_w = a_v = np.zeros((N_DOF, n_patch), np.float32)

    for h in handles:
        h.remove()
    del vla, processor
    torch.cuda.empty_cache()

    rec = dict(
        name=name, text_w=t_w, text_v=t_v, act_w=a_w, act_v=a_v, act_ok=act_ok,
        text_H_w=entropy_rows(t_w), text_H_v=entropy_rows(t_v),
        text_fn_w=fnorm_rows(t_w), text_fn_v=fnorm_rows(t_v),
        text_peak_v=peak(t_v),
        act_H_v=entropy_rows(a_v) if act_ok else np.nan,
        act_fn_v=fnorm_rows(a_v) if act_ok else np.nan,
        act_peak_v=peak(a_v) if act_ok else np.nan,
    )
    print(f"  text H_v={rec['text_H_v']:.3f} fn_v={rec['text_fn_v']:.4f} peak={rec['text_peak_v']:.3f} "
          f"| act H_v={rec['act_H_v']}", flush=True)
    return rec


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    os.makedirs(OUT, exist_ok=True)

    d = np.load(SCENE_NPZ, allow_pickle=True)
    img_clean = d["clean_rgb"][0]
    img_poison = d["poison_rgb"][0]
    prompt = str(d["prompt"])
    # extract task phrase
    desc = "turn on the stove"
    print(f"prompt={prompt!r}", flush=True)

    # Conditions: each model × matching scene
    # Base VLM & clean VLA see CLEAN scene; GoBA bd sees POISON (trigger) scene —
    # also run base/clean on poison scene as control (trigger without backdoor training).
    jobs = [
        (BASE_VLM, img_clean, "BASE_VLM×clean_scene"),
        (BASE_VLM, img_poison, "BASE_VLM×trigger_scene"),
        (CLEAN_VLA, img_clean, "CLEAN_VLA×clean_scene"),
        (CLEAN_VLA, img_poison, "CLEAN_VLA×trigger_scene"),
        (GOBA_CKPT, img_clean, "GOBA_BD×clean_scene"),
        (GOBA_CKPT, img_poison, "GOBA_BD×trigger_scene"),
    ]
    results = [probe(ckpt, img, desc, name) for ckpt, img, name in jobs]

    # Save
    np.savez_compressed(
        f"{OUT}/compare.npz",
        clean_rgb=img_clean, poison_rgb=img_poison,
        names=np.array([r["name"] for r in results], dtype=object),
        text_H_v=np.array([r["text_H_v"] for r in results]),
        text_fn_v=np.array([r["text_fn_v"] for r in results]),
        text_peak_v=np.array([r["text_peak_v"] for r in results]),
        act_H_v=np.array([r["act_H_v"] for r in results]),
        act_fn_v=np.array([r["act_fn_v"] for r in results]),
        **{f"{r['name']}_text_v": r["text_v"] for r in results},
        **{f"{r['name']}_act_v": r["act_v"] for r in results},
        prompt=np.array(prompt),
    )

    # Bar chart of stats
    names = [r["name"].replace("×", "\n") for r in results]
    x = np.arange(len(results))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, key, title in [
        (axes[0], "text_H_v", "TEXT→img entropy (α·‖Wov‖)"),
        (axes[1], "text_fn_v", "TEXT→img f_norm"),
        (axes[2], "act_H_v", "ACTION→img entropy"),
    ]:
        vals = [r[key] for r in results]
        colors = []
        for n in [r["name"] for r in results]:
            if "BASE" in n: colors.append("seagreen")
            elif "CLEAN_VLA" in n: colors.append("steelblue")
            else: colors.append("darkorange")
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
        ax.set_title(title); ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Same scene: base VLM vs clean VLA vs GoBA backdoor\n"
                 "(green=base OpenVLA-7B, blue=clean LIBERO, orange=GoBA bd)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/BARS_stats_compare.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Overlay grid: text mean map for key conditions
    pick = [
        ("BASE_VLM×clean_scene", img_clean),
        ("BASE_VLM×trigger_scene", img_poison),
        ("CLEAN_VLA×clean_scene", img_clean),
        ("CLEAN_VLA×trigger_scene", img_poison),
        ("GOBA_BD×clean_scene", img_clean),
        ("GOBA_BD×trigger_scene", img_poison),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    by_name = {r["name"]: r for r in results}
    for ax, (name, rgb) in zip(axes.ravel(), pick):
        M = by_name[name]["text_v"].mean(0)
        up = zoom(M.reshape(GRID, GRID), rgb.shape[0] / GRID, order=1)
        ax.imshow(rgb); ax.imshow(up, cmap="jet", alpha=0.55)
        ax.set_title(f"{name}\nH={by_name[name]['text_H_v']:.2f} peak={by_name[name]['text_peak_v']:.2f}",
                     fontsize=8)
        ax.axis("off")
    fig.suptitle("TEXT→img value-weighted overlays (same prompt, last layer)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/OVERLAY_text_v_all_conditions.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Action overlays
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, (name, rgb) in zip(axes.ravel(), pick):
        r = by_name[name]
        if not r["act_ok"]:
            ax.set_title(f"{name}\n(no action)"); ax.axis("off"); continue
        M = r["act_v"].mean(0)
        up = zoom(M.reshape(GRID, GRID), rgb.shape[0] / GRID, order=1)
        ax.imshow(rgb); ax.imshow(up, cmap="jet", alpha=0.55)
        ax.set_title(f"{name}\nH={r['act_H_v']:.2f}", fontsize=8); ax.axis("off")
    fig.suptitle("ACTION→img value-weighted overlays")
    fig.tight_layout()
    fig.savefig(f"{OUT}/OVERLAY_action_v_all_conditions.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    lines = ["VLM vs VLA attention compare (last layer, α·‖Wov‖)\n"]
    lines.append(f"{'condition':32s} {'H_text':>7} {'fn_text':>8} {'peak':>7} {'H_act':>7}")
    for r in results:
        lines.append(f"{r['name']:32s} {r['text_H_v']:7.3f} {r['text_fn_v']:8.4f} "
                     f"{r['text_peak_v']:7.3f} {r['act_H_v']:7.3f}")
    lines.append("\nInterpretation guide:")
    lines.append("- BASE_VLM = openvla/openvla-7b (pretrained, no LIBERO / no backdoor)")
    lines.append("- CLEAN_VLA = openvla-7b-finetuned-libero-goal")
    lines.append("- GOBA_BD = your backdoored checkpoint")
    lines.append("- trigger_scene has the mug; clean_scene does not")
    lines.append("- If BASE/CLEAN on trigger_scene do NOT match GOBA_BD×trigger,")
    lines.append("  the signature needs backdoor training (not just the visual mug).")
    with open(f"{OUT}/SUMMARY.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nDone -> {OUT}")


if __name__ == "__main__":
    main()
