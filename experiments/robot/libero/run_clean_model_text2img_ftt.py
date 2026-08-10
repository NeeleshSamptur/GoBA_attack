"""
Clean OpenVLA control: text→image FTT (T2IShield) on clean vs poison BDDL.
Same extraction as run_action_query_ftt_goba.py text2img path.
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

CHECKPOINT = "openvla/openvla-7b-finetuned-libero-goal"
OUT = f"{REPO}/attn_maps/infoentropy_ieattn/clean_model_text2img_ftt.npz"
BDDL = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
SUITE = "libero_goal"
SEED = 7
NUM_STEPS_WAIT = 10
DEVICE = 0
N_DOF = 7


def ftt(rows):
    p = rows / np.clip(rows.sum(axis=1, keepdims=True), 1e-12, None)
    return float(np.linalg.norm(p - p.mean(axis=0, keepdims=True), axis=1).mean())


def preprocess(img):
    im = tf.convert_to_tensor(np.array(Image.fromarray(img).convert("RGB")))
    dt = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, dt, saturate=True)
    return Image.fromarray(im.numpy()).convert("RGB")


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    print("Loading CLEAN OpenVLA...", flush=True)
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        CHECKPOINT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE).eval()
    stats_path = hf_hub_download(CHECKPOINT, "dataset_statistics.json")
    vla.norm_stats = json.load(open(stats_path))
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches

    def text2img_ftt(image, desc):
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
        input_ids = inputs.input_ids
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)
        n_lang = input_ids.shape[-1] - 1
        t0 = 1 + num_patches
        t1 = t0 + n_lang
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = vla.generate(
                input_ids, pixel_values=inputs.pixel_values,
                max_new_tokens=N_DOF, output_attentions=True,
                return_dict_in_generate=True, do_sample=False,
            )
        n_layers = len(gen.attentions[0])
        pre = gen.attentions[0]
        out = np.full(n_layers, np.nan)
        for l in range(n_layers):
            la = pre[l][0].float().mean(dim=0)
            if la.dim() == 2 and la.shape[0] > 1:
                rows = la[t0:t1, 1:1 + num_patches].cpu().numpy()
                out[l] = ftt(rows)
        del gen
        torch.cuda.empty_cache()
        return out, n_lang

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
            layers, n_lang = text2img_ftt(img, desc)
            records.append(dict(task_id=task_id, cond=cond, seed=SEED, n_lang=n_lang, text2img=layers))
            print(f"CLEAN_MODEL task={task_id} {cond:6s} n_lang={n_lang}  "
                  f"L31 t2i_FTT={layers[-1]:.4f}", flush=True)

    text2img = np.stack([r["text2img"] for r in records])
    cond = np.array([r["cond"] for r in records])
    task_id = np.array([r["task_id"] for r in records])
    np.savez_compressed(
        OUT, task_id=task_id, cond=cond, seed=np.array([r["seed"] for r in records]),
        n_lang=np.array([r["n_lang"] for r in records]), text2img=text2img,
    )

    # L31 AUROC vs GoBA reference
    s = text2img[:, -1]
    y = (cond == "poison").astype(int)
    print("\n=== Clean model text→image FTT (L31) ===")
    print(f"clean mean={s[cond=='clean'].mean():.5f}  poison mean={s[cond=='poison'].mean():.5f}")
    print(f"AUROC poison↓={roc_auc_score(y, -s):.3f}  poison↑={roc_auc_score(y, s):.3f}")
    for t in range(10):
        c = s[(task_id == t) & (cond == "clean")][0]
        p = s[(task_id == t) & (cond == "poison")][0]
        print(f"  t{t}: clean={c:.5f} poison={p:.5f} Δ(c-p)={c-p:.5f}")
    deltas = [s[(task_id == t) & (cond == "clean")][0] - s[(task_id == t) & (cond == "poison")][0]
              for t in range(10)]
    print(f"paired Δ>0: {sum(d > 0 for d in deltas)}/10")

    goba = np.load(f"{REPO}/attn_maps/goba_action_query_ftt.npz", allow_pickle=True)
    gr = np.array([str(r) for r in goba["role"]])
    gt = goba["task_id"]
    # mean over seeds per task for fair-ish compare
    gc = np.array([goba["text2img"][(gt == t) & (gr == "clean")][:, -1].mean() for t in range(10)])
    gp = np.array([goba["text2img"][(gt == t) & (gr == "poison")][:, -1].mean() for t in range(10)])
    ys = np.array([0] * 10 + [1] * 10)
    ss = np.concatenate([gc, gp])
    print("\n=== GoBA text→image FTT (L31, mean over seeds) for reference ===")
    print(f"clean mean={gc.mean():.5f}  poison mean={gp.mean():.5f}")
    print(f"AUROC poison↓={roc_auc_score(ys, -ss):.3f}")
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
