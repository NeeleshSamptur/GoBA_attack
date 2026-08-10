"""
Calibration-free KL assimilation on text→image rows (like FTT, no clean ref set).

score = mean_i KL(p_i || p_bar),  p_bar = mean row after row-normalize.
Also JS for reference. L31 head-avg.
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
from huggingface_hub import hf_hub_download
from libero.libero import benchmark
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.libero.run_libero_eval_attentionmap import crop_and_resize
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

OUT = f"{REPO}/attn_maps/infoentropy_ieattn"
BDDL = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
SUITE = "libero_goal"
SEED = 7
NUM_STEPS_WAIT = 10
DEVICE = 0
N_DOF = 7
EPS = 1e-12

GOBA_CKPT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
CLEAN_CKPT = "openvla/openvla-7b-finetuned-libero-goal"


def preprocess(img):
    im = tf.convert_to_tensor(np.array(Image.fromarray(img).convert("RGB")))
    dt = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, dt, saturate=True)
    return Image.fromarray(im.numpy()).convert("RGB")


def row_norm(P):
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(1, keepdims=True), EPS, None)


def ftt(P):
    P = row_norm(P)
    return float(np.linalg.norm(P - P.mean(0), axis=1).mean())


def mean_kl_to_mean(P):
    """Calibration-free: mean_i KL(p_i || p_bar)."""
    P = row_norm(P)
    bar = np.clip(P.mean(0), EPS, None)
    bar = bar / bar.sum()
    P = np.clip(P, EPS, None)
    kl = (P * (np.log(P) - np.log(bar))).sum(1)
    return float(kl.mean())


def mean_js_to_mean(P):
    P = row_norm(P)
    bar = P.mean(0)
    js = []
    for i in range(P.shape[0]):
        m = 0.5 * (P[i] + bar)
        m = np.clip(m, EPS, None)
        pi = np.clip(P[i], EPS, None)
        b = np.clip(bar, EPS, None)
        js.append(0.5 * (pi * np.log(pi / m)).sum() + 0.5 * (b * np.log(b / m)).sum())
    return float(np.mean(js))


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


def text2img_rows(vla, processor, image, desc):
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    input_ids = inputs.input_ids
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)
    n_lang = input_ids.shape[-1] - 1
    t0, t1 = 1 + num_patches, 1 + num_patches + n_lang
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        gen = vla.generate(
            input_ids, pixel_values=inputs.pixel_values,
            max_new_tokens=N_DOF, output_attentions=True,
            return_dict_in_generate=True, do_sample=False,
        )
    pre = gen.attentions[0]
    la = pre[-1][0].float().mean(0)  # last layer, head-avg
    rows = la[t0:t1, 1:1 + num_patches].cpu().numpy()
    del gen
    torch.cuda.empty_cache()
    return rows


def run(name, ckpt):
    print(f"\n=== {name} ===", flush=True)
    processor, vla = load_model(ckpt)
    suite = benchmark.get_benchmark_dict()[SUITE]()
    recs = []
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
            P = text2img_rows(vla, processor, img, desc)
            rec = dict(
                task_id=task_id, cond=cond,
                ftt=ftt(P), kl=mean_kl_to_mean(P), js=mean_js_to_mean(P),
            )
            recs.append(rec)
            print(f"{name} t={task_id} {cond:6s} FTT={rec['ftt']:.5f} KL={rec['kl']:.5f} JS={rec['js']:.5f}",
                  flush=True)
    del vla, processor
    torch.cuda.empty_cache()
    return recs


def summarize(recs, name):
    cond = np.array([r["cond"] for r in recs])
    tid = np.array([r["task_id"] for r in recs])
    y = (cond == "poison").astype(int)
    lines = [f"## {name}", ""]
    lines.append("| metric | AUROC | clean | poison | poison↓? | paired clean>poison |")
    lines.append("|---|---:|---:|---:|---|---:|")
    for key, label in [("ftt", "FTT"), ("kl", "mean KL(p_i‖p̄)"), ("js", "mean JS(p_i,p̄)")]:
        s = np.array([r[key] for r in recs], float)
        a_lo = roc_auc_score(y, -s)
        a_hi = roc_auc_score(y, s)
        auc, higher = (a_lo, False) if a_lo >= a_hi else (a_hi, True)
        cm, pm = float(s[cond == "clean"].mean()), float(s[cond == "poison"].mean())
        npos = sum(s[(tid == t) & (cond == "clean")][0] > s[(tid == t) & (cond == "poison")][0]
                   for t in range(10))
        lines.append(
            f"| {label} | {auc:.3f} | {cm:.5f} | {pm:.5f} | "
            f"{'yes' if not higher else 'no'} | {npos}/10 |"
        )
        print(f"{name:14s} {label:20s} AUROC={auc:.3f} clean={cm:.5f} poison={pm:.5f} "
              f"poison↓={not higher} paired_c>p={npos}/10")
    lines.append("")
    return "\n".join(lines)


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    os.makedirs(OUT, exist_ok=True)

    md = [
        "# Calibration-free KL vs FTT (text→image rows, L31)",
        "",
        "No clean reference set. Within each sample:",
        "`score = mean_i KL(p_i || p̄)` with `p̄ = mean_i p_i` after row-sum normalize.",
        "Same for JS. Seed=7, tasks 0–9, clean vs poison BDDL.",
        "",
    ]
    save = {}
    for name, ckpt in [("goba", GOBA_CKPT), ("clean_openvla", CLEAN_CKPT)]:
        recs = run(name, ckpt)
        md.append(summarize(recs, name))
        save[f"{name}_task_id"] = np.array([r["task_id"] for r in recs])
        save[f"{name}_cond"] = np.array([r["cond"] for r in recs])
        for k in ("ftt", "kl", "js"):
            save[f"{name}_{k}"] = np.array([r[k] for r in recs])

    md += [
        "## Reading",
        "- If KL tracks FTT (high AUROC on GoBA, ~0.5 on clean), KL is a valid FTT cousin.",
        "- Both should drop under poison (rows closer to mean → smaller KL to mean).",
        "",
    ]
    np.savez_compressed(f"{OUT}/kl_vs_ftt_goba_family.npz", **save)
    with open(f"{OUT}/SUMMARY_kl_vs_ftt.md", "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(f"Saved {OUT}/kl_vs_ftt_goba_family.npz", flush=True)


if __name__ == "__main__":
    main()
