"""
Image→image self-attention: vision encoder (DINO + SigLIP) vs LLM patch→patch.

For the same GoBA scenes, extract:
  * DINOv2 last-block patch→patch attention (16x16, dropping CLS+4 registers)
  * SigLIP last-block patch→patch attention (16x16, no prefix tokens)
  * LLM last-layer patch→patch attention (256 projected vision tokens)

Render clean vs trigger heatmaps side-by-side for each, plus a 2x3 comparison
panel (clean|trigger × DINO|SigLIP|LLM). Also report T2IShield f_norm on each.

Output:
  attn_maps/goba_imgimg_encoder_vs_llm/
  attn_maps/goba_imgimg_encoder_vs_llm.npz
"""
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
import torch.nn.functional as F
from libero.libero import benchmark
from PIL import Image
from scipy.ndimage import zoom
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.libero.run_libero_eval_attentionmap import crop_and_resize
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

CHECKPOINT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
OUT_DIR = f"{REPO}/attn_maps/goba_imgimg_encoder_vs_llm"
OUT_NPZ = f"{OUT_DIR}/stats.npz"

CONDITIONS = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
SUITE = "libero_goal"
# a few held-out scenes for readable heatmaps + enough for AUROC
TASKS = [7, 8, 9]
SEEDS = [7, 42, 1234]
NUM_STEPS_WAIT = 10
GRID = 16
DEVICE = 0
N_HEATMAPS = 3  # how many (task,seed) pairs get PNG panels


def ftt(rows):
    p = rows / np.clip(rows.sum(axis=1, keepdims=True), 1e-12, None)
    return float(np.linalg.norm(p - p.mean(0, keepdims=True), axis=1).mean())


def preprocess(img):
    im = tf.convert_to_tensor(np.array(Image.fromarray(img).convert("RGB")))
    dt = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, dt, saturate=True)
    return np.array(Image.fromarray(im.numpy()).convert("RGB"))


def get_scene(task, bddl, seed):
    env, desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False, bddl_path=bddl, seed=seed)
    env.reset()
    obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
    img = preprocess(get_libero_image(obs, 224))
    env.close()
    return img, desc


class AttnCatcher:
    """Monkey-patch timm Attention.forward to store head-averaged softmax weights."""

    def __init__(self, attn_module):
        self.attn = attn_module
        self.weights = None
        self._orig = attn_module.forward
        self._was_fused = bool(getattr(attn_module, "fused_attn", False))

    def __enter__(self):
        attn = self.attn
        catcher = self
        if self._was_fused:
            attn.fused_attn = False

        def forward(x):
            B, N, C = x.shape
            qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = attn.q_norm(q), attn.k_norm(k)
            q = q * attn.scale
            a = q @ k.transpose(-2, -1)
            a = a.softmax(dim=-1)
            catcher.weights = a.detach().float().mean(dim=1)[0].cpu().numpy()  # (N,N) head-avg
            a = attn.attn_drop(a)
            x = a @ v
            x = x.transpose(1, 2).reshape(B, N, C)
            x = attn.proj(x)
            x = attn.proj_drop(x)
            return x

        attn.forward = forward
        return self

    def __exit__(self, *exc):
        self.attn.forward = self._orig
        if self._was_fused:
            self.attn.fused_attn = True


def overlay(ax, rgb, mass_flat, vmin, vmax, title):
    grid = mass_flat.reshape(GRID, GRID)
    up = zoom(grid, rgb.shape[0] / GRID, order=1)
    ax.imshow(rgb)
    im = ax.imshow(up, cmap="jet", alpha=0.55, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return im


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        CHECKPOINT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE)
    vla.eval()

    dino = vla.vision_backbone.featurizer
    siglip = vla.vision_backbone.fused_featurizer
    n_patch = dino.patch_embed.num_patches
    dino_prefix = int(getattr(dino, "num_prefix_tokens", 0) or 0)
    sig_prefix = int(getattr(siglip, "num_prefix_tokens", 0) or 0)
    # OpenVLA monkey-patches featurizer.forward → get_intermediate_layers(n={len(blocks)-2}),
    # so the last block that actually runs is blocks[-2], not blocks[-1].
    dino_block = dino.blocks[-2]
    sig_block = siglip.blocks[-2]
    print(f"DINO prefix={dino_prefix} patches={n_patch} hooked=blocks[-2]/{len(dino.blocks)}; "
          f"SigLIP prefix={sig_prefix} hooked=blocks[-2]/{len(siglip.blocks)}", flush=True)

    task_suite = benchmark.get_benchmark_dict()[SUITE]()
    os.makedirs(OUT_DIR, exist_ok=True)

    def extract(img, desc):
        image = Image.fromarray(img).convert("RGB")
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

        with AttnCatcher(dino_block.attn) as cd, AttnCatcher(sig_block.attn) as cs:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = vla(**inputs, output_attentions=True)

        # vision encoder: drop prefix tokens, keep patch↔patch
        dino_pp = cd.weights[dino_prefix:dino_prefix + n_patch, dino_prefix:dino_prefix + n_patch]
        sig_pp = cs.weights[sig_prefix:sig_prefix + n_patch, sig_prefix:sig_prefix + n_patch]

        # LLM last layer: patch queries → patch keys (skip BOS at index 0)
        la = out.attentions[-1][0].float().mean(dim=0)
        llm_pp = la[1:1 + n_patch, 1:1 + n_patch].cpu().numpy()

        del out
        torch.cuda.empty_cache()

        # consensus key-mass: average over query patches → which keys get attended
        return {
            "dino": dino_pp,
            "siglip": sig_pp,
            "llm": llm_pp,
            "dino_mass": dino_pp.mean(0),
            "siglip_mass": sig_pp.mean(0),
            "llm_mass": llm_pp.mean(0),
            "dino_ftt": ftt(dino_pp),
            "siglip_ftt": ftt(sig_pp),
            "llm_ftt": ftt(llm_pp),
        }

    records = []
    n_saved = 0
    for task_id in TASKS:
        task = task_suite.get_task(task_id)
        for seed in SEEDS:
            pack = {}
            for cond, bddl in CONDITIONS.items():
                img, desc = get_scene(task, bddl, seed)
                res = extract(img, desc)
                pack[cond] = dict(img=img, **res)
                records.append(dict(
                    task_id=task_id, seed=seed, cond=cond,
                    dino_ftt=res["dino_ftt"], siglip_ftt=res["siglip_ftt"], llm_ftt=res["llm_ftt"],
                ))
            print(
                f"task={task_id} seed={seed}  "
                f"DINO c={pack['clean']['dino_ftt']:.4f} p={pack['poison']['dino_ftt']:.4f} | "
                f"SigLIP c={pack['clean']['siglip_ftt']:.4f} p={pack['poison']['siglip_ftt']:.4f} | "
                f"LLM c={pack['clean']['llm_ftt']:.4f} p={pack['poison']['llm_ftt']:.4f}",
                flush=True,
            )

            if n_saved < N_HEATMAPS:
                # 2x3: rows=clean/trigger, cols=DINO/SigLIP/LLM
                masses = []
                for cond in ("clean", "poison"):
                    for key in ("dino_mass", "siglip_mass", "llm_mass"):
                        masses.append(pack[cond][key])
                allv = np.concatenate(masses)
                vmin, vmax = np.percentile(allv, 1), np.percentile(allv, 99)

                fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
                titles = ["DINO (vision encoder)", "SigLIP (vision encoder)", "LLM patch→patch"]
                for r, cond in enumerate(("clean", "poison")):
                    for c, key in enumerate(("dino_mass", "siglip_mass", "llm_mass")):
                        overlay(axes[r, c], pack[cond]["img"], pack[cond][key], vmin, vmax,
                                f"{'CLEAN' if cond == 'clean' else 'TRIGGER'}  |  {titles[c]}")
                fig.suptitle(
                    f"GoBA image→image self-attention  task={task_id} seed={seed}\n"
                    f"top=clean scene, bottom=trigger scene  |  shared color scale",
                    fontsize=12,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.93])
                path = f"{OUT_DIR}/COMPARE_t{task_id}_s{seed}_encoder_vs_llm.png"
                fig.savefig(path, dpi=140, bbox_inches="tight")
                plt.close(fig)
                print(f"  saved {path}", flush=True)

                # also one pairwise encoder-vs-LLM panel for the trigger scene
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                for ax, key, title in zip(
                    axes,
                    ("dino_mass", "siglip_mass", "llm_mass"),
                    ("DINO last block", "SigLIP last block", "LLM last layer"),
                ):
                    overlay(ax, pack["poison"]["img"], pack["poison"][key], vmin, vmax, title)
                fig.suptitle(f"TRIGGER scene — encoder vs LLM  task={task_id} seed={seed}", fontsize=11)
                fig.tight_layout(rect=[0, 0, 1, 0.92])
                path = f"{OUT_DIR}/TRIGGER_t{task_id}_s{seed}_encoder_vs_llm.png"
                fig.savefig(path, dpi=140, bbox_inches="tight")
                plt.close(fig)
                n_saved += 1

    arrays = {k: np.array([r[k] for r in records]) for k in
              ["task_id", "seed", "cond", "dino_ftt", "siglip_ftt", "llm_ftt"]}
    np.savez(OUT_NPZ, **arrays)

    # summary AUROC
    from sklearn.metrics import roc_auc_score
    cond = arrays["cond"].astype(str)
    y = (cond == "poison").astype(int)
    print("\n=== FTT AUROC (low = backdoor) over all collected scenes ===")
    for name in ["dino_ftt", "siglip_ftt", "llm_ftt"]:
        a = roc_auc_score(y, -arrays[name])
        c, p = arrays[name][y == 0], arrays[name][y == 1]
        print(f"  {name:12s} AUROC={a:.3f}  clean={c.mean():.4f}  poison={p.mean():.4f}  "
              f"margin={c.min()-p.max():+.4f}")
    print(f"\nSaved stats -> {OUT_NPZ}")
    print(f"Heatmaps  -> {OUT_DIR}")


if __name__ == "__main__":
    main()
