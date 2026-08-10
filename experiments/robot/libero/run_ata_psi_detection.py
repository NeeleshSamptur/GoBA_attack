"""
ATA-style Ψ detection: last-query → image attention map scores.
GoBA-backdoored + clean OpenVLA control, clean vs poison BDDL, seed=7.
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
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_auc_score
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.libero.run_libero_eval_attentionmap import crop_and_resize
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

GOBA_CKPT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
CLEAN_CKPT = "openvla/openvla-7b-finetuned-libero-goal"
OUT = f"{REPO}/attn_maps/ata_psi_detection"
BDDL = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
SUITE = "libero_goal"
SEED = 7
NUM_STEPS_WAIT = 10
DEVICE = 0
N_DOF = 7
LAYER = -1  # last LLM layer


def preprocess(img):
    im = tf.convert_to_tensor(np.array(Image.fromarray(img).convert("RGB")))
    dt = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, dt, saturate=True)
    return Image.fromarray(im.numpy()).convert("RGB")


def score_map(psi):
    """psi: (256,) nonnegative. Return dict of scalar scores."""
    p = np.clip(psi, 0, None)
    p = p / np.clip(p.sum(), 1e-12, None)
    H = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    return dict(
        max=float(p.max()),
        top4=float(np.sort(p)[-4:].sum()),
        entropy=H,  # high = flat
        inv_entropy=1.0 - H,  # high = peaked
    )


def ftt_rows(rows):
    p = rows / np.clip(rows.sum(1, keepdims=True), 1e-12, None)
    return float(np.linalg.norm(p - p.mean(0, keepdims=True), axis=1).mean())


def load_model(ckpt):
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE).eval()
    if os.path.isdir(ckpt) and os.path.exists(os.path.join(ckpt, "dataset_statistics.json")):
        vla.norm_stats = json.load(open(os.path.join(ckpt, "dataset_statistics.json")))
    else:
        vla.norm_stats = json.load(open(hf_hub_download(ckpt, "dataset_statistics.json")))
    return processor, vla


def extract_psi(vla, processor, image, desc):
    """Return last-query→image Ψ, action-mean→image, action-row FTT at LAYER."""
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    input_ids = inputs.input_ids
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        gen = vla.generate(
            input_ids, pixel_values=inputs.pixel_values,
            max_new_tokens=N_DOF, output_attentions=True,
            return_dict_in_generate=True, do_sample=False,
        )
    # Prefer full-seq forward for clear last-token row (matches ATA "last query")
    seq = gen.sequences
    del gen
    torch.cuda.empty_cache()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(input_ids=seq, pixel_values=inputs.pixel_values,
                  output_attentions=True, return_dict=True)
    n_layers = len(out.attentions)
    l = n_layers + LAYER if LAYER < 0 else LAYER
    A = out.attentions[l][0].float().mean(0)  # (N, N) head-avg
    N = A.shape[-1]
    # last query
    row_last = A[-1, 1:1 + num_patches].cpu().numpy()
    # 7 action queries
    rows_act = A[-N_DOF:, 1:1 + num_patches].cpu().numpy()
    psi_last = row_last / np.clip(row_last.sum(), 1e-12, None)
    psi_act = rows_act.mean(0)
    psi_act = psi_act / np.clip(psi_act.sum(), 1e-12, None)
    sc_last = score_map(psi_last)
    sc_act = score_map(psi_act)
    sc_act["ftt"] = ftt_rows(rows_act)
    del out
    torch.cuda.empty_cache()
    return psi_last.astype(np.float32), psi_act.astype(np.float32), sc_last, sc_act


def run_model(name, ckpt):
    print(f"\n=== {name} ===", flush=True)
    processor, vla = load_model(ckpt)
    suite = benchmark.get_benchmark_dict()[SUITE]()
    records = []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        for cond, bddl in BDDL.items():
            env, desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False,
                                       bddl_path=bddl, seed=SEED)
            env.reset()
            obs = None
            for _ in range(NUM_STEPS_WAIT):
                obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
            env.close()
            img = preprocess(get_libero_image(obs, 224))
            psi_l, psi_a, sc_l, sc_a = extract_psi(vla, processor, img, desc)
            rec = dict(task_id=task_id, cond=cond, seed=SEED,
                       psi_last=psi_l, psi_actmean=psi_a, **{f"last_{k}": v for k, v in sc_l.items()},
                       **{f"act_{k}": v for k, v in sc_a.items()})
            records.append(rec)
            print(f"{name} task={task_id} {cond:6s}  "
                  f"last_max={sc_l['max']:.4f} last_H={sc_l['entropy']:.4f}  "
                  f"act_max={sc_a['max']:.4f} act_FTT={sc_a['ftt']:.4f}", flush=True)
    del vla, processor
    torch.cuda.empty_cache()
    return records


def summarize(records, name):
    cond = np.array([r["cond"] for r in records])
    tid = np.array([r["task_id"] for r in records])
    y = (cond == "poison").astype(int)
    keys = [
        ("last_max", True, "last-query max(Ψ)"),
        ("last_top4", True, "last-query top4(Ψ)"),
        ("last_inv_entropy", True, "last-query 1−H(Ψ)"),
        ("last_entropy", False, "last-query H(Ψ)"),
        ("act_max", True, "action-mean max(Ψ)"),
        ("act_top4", True, "action-mean top4(Ψ)"),
        ("act_inv_entropy", True, "action-mean 1−H(Ψ)"),
        ("act_ftt", False, "action-rows FTT"),  # poison often lower
    ]
    lines = [f"## {name}", ""]
    lines.append("| metric | AUROC | clean | poison | poison↑? | paired Δ>0 |")
    lines.append("|---|---:|---:|---:|---|---:|")
    print(f"\n--- {name} AUROC ---")
    best = None
    for key, prefer_hi, label in keys:
        s = np.array([r[key] for r in records], float)
        a_hi = roc_auc_score(y, s)
        a_lo = roc_auc_score(y, -s)
        if prefer_hi:
            # report best but note convention: for peakiness expect poison↑
            auc, higher = (a_hi, True) if a_hi >= a_lo else (a_lo, False)
        else:
            auc, higher = (a_lo, False) if a_lo >= a_hi else (a_hi, True)
        # always pick best orientation for honesty
        if a_hi >= a_lo:
            auc, higher = a_hi, True
        else:
            auc, higher = a_lo, False
        cm = float(s[cond == "clean"].mean())
        pm = float(s[cond == "poison"].mean())
        deltas = []
        for t in range(10):
            c = s[(tid == t) & (cond == "clean")][0]
            p = s[(tid == t) & (cond == "poison")][0]
            deltas.append(c - p if not higher else p - c)
        # paired: count tasks where poison is more "backdoor-like"
        if higher:
            npos = sum(
                s[(tid == t) & (cond == "poison")][0] > s[(tid == t) & (cond == "clean")][0]
                for t in range(10))
        else:
            npos = sum(
                s[(tid == t) & (cond == "poison")][0] < s[(tid == t) & (cond == "clean")][0]
                for t in range(10))
        lines.append(
            f"| {label} | {auc:.3f} | {cm:.4f} | {pm:.4f} | "
            f"{'yes' if higher else 'no'} | {npos}/10 |"
        )
        print(f"  {label:28s} AUROC={auc:.3f}  clean={cm:.4f} poison={pm:.4f}  "
              f"poison↑={higher}  paired={npos}/10")
        if best is None or auc > best[0]:
            best = (auc, label, higher)
    lines.append("")
    return "\n".join(lines), records


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    os.makedirs(OUT, exist_ok=True)

    all_md = [
        "# ATA-style Ψ (last-query→image) as detector",
        "",
        "Protocol: seed=7, tasks 0–9, clean vs poison BDDL, last LLM layer, head-avg.",
        "Ψ = attention from last query token to 256 image patches (ATA Sec. III-B).",
        "Also: mean of 7 action-query rows + FTT on those rows.",
        "",
    ]
    save = {}
    for name, ckpt in [("goba", GOBA_CKPT), ("clean_model", CLEAN_CKPT)]:
        recs = run_model(name, ckpt)
        md, recs = summarize(recs, name)
        all_md.append(md)
        save[f"{name}_task_id"] = np.array([r["task_id"] for r in recs])
        save[f"{name}_cond"] = np.array([r["cond"] for r in recs])
        for k in recs[0]:
            if k in ("task_id", "cond", "seed"):
                continue
            if isinstance(recs[0][k], np.ndarray):
                save[f"{name}_{k}"] = np.stack([r[k] for r in recs])
            else:
                save[f"{name}_{k}"] = np.array([r[k] for r in recs])

    all_md += [
        "## Reading",
        "",
        "- Strong detector: high AUROC on **GoBA**, ~0.5 on **clean_model** (not just sticker).",
        "- If both high in same direction → scene confound (like Ieattn).",
        "",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(all_md))
    np.savez_compressed(f"{OUT}/psi_scores.npz", **save)
    print(f"\nWrote {OUT}/SUMMARY.md", flush=True)
    print("\n".join(all_md))


if __name__ == "__main__":
    main()
