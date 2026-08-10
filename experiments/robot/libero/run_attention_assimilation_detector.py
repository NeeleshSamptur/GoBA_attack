"""
Training-free backdoor detection from cross-token attention structure.

Adapts T2IShield's "assimilation" idea (arXiv 2407.04215) to GoBA/OpenVLA:
when the physical trigger is in frame, the text tokens' attention-over-patches
maps collapse toward each other, so their dispersion around the mean map drops.

T2IShield FTT (their Eq.):   F = (1/L) sum_i || M^(i) - Mbar ||_F ,  backdoor if F < threshold
Here M^(i) is text-token i's attention over the 16x16 patch grid (row-normalized
to a distribution), Mbar is the mean map across the L text tokens.

Also computes several sibling statistics on the same forward pass so they can be
compared on equal footing, at every LLM layer:

  f_norm            T2IShield FTT (cross-token dispersion)      expect DOWN w/ trigger
  mean_pairwise_cos cross-token cosine similarity               expect UP   w/ trigger
  consensus_entropy entropy of the token-averaged map           expect DOWN w/ trigger
  mean_token_entropy avg within-token map entropy               expect DOWN w/ trigger
  mean_token_max    avg within-token peak mass                  expect UP   w/ trigger

None of these require knowing where the trigger is, or any training.
`trig_mass` is also logged but is an ORACLE (uses a hardcoded trigger box) and is
only there as a sanity reference -- it is not a deployable detector.

Outputs a .npz of per-sample per-layer statistics; analyze with the companion
AUROC script.
"""
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
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
)
# Reuse GoBA authors' helpers from run_libero_eval_attentionmap.py as-is;
# this file only adds assimilation stats (f_norm, etc.) + clean/poison protocol.
from experiments.robot.libero.run_libero_eval_attentionmap import (
    crop_and_resize,
    get_avg_patch_text_attention,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer

CHECKPOINT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
OUT_NPZ = f"{REPO}/attn_maps/assimilation_stats_alllayers_fnorm_disjoint.npz"

CONDITIONS = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}

SUITE = "libero_goal"
# NOTE: LIBERO-Goal shares one scene layout across all 10 tasks (tasks differ only by
# instruction), so visual diversity comes ONLY from the seed. More seeds = more scenes.
#
# DISJOINT sets, deliberately: clean and poison must NEVER use the same seed, since
# env.seed(seed) controls object placement in the sim -- reusing a seed across
# conditions would pair each clean scene with an (almost) identical poison scene that
# only differs by whether the trigger object was inserted, which is a paired-comparison
# design, not the independent clean-vs-poison sample sets we actually want here.
CLEAN_SEEDS = [7, 42, 1234, 2026, 31337, 5, 99, 777, 20260803, 424242]
POISON_SEEDS = [11, 43, 1337, 2027, 31338, 6, 100, 778, 20260804, 424243]
assert not set(CLEAN_SEEDS) & set(POISON_SEEDS), "clean/poison seed sets must be disjoint"
NUM_STEPS_WAIT = 10
DEVICE = 0


def compute_fnorm(t2p):
    """t2p: [L_tokens, 256] raw attention. Returns only f_norm."""
    p = t2p / np.clip(t2p.sum(axis=1, keepdims=True), 1e-8, None)
    mbar = p.mean(axis=0)
    f_norm = np.linalg.norm(p - mbar[None, :], axis=1).mean()
    return float(f_norm)


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
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches
    print(f"Model loaded. num_patches={num_patches}", flush=True)

    task_suite = benchmark.get_benchmark_dict()[SUITE]()
    n_tasks = task_suite.n_tasks

    records = []

    for task_id in range(n_tasks):
        task = task_suite.get_task(task_id)
        for cond, bddl_dir in CONDITIONS.items():
            seeds_for_cond = CLEAN_SEEDS if cond == "clean" else POISON_SEEDS
            for seed in seeds_for_cond:
                env, desc = get_libero_env(
                    task, "openvla", resolution=256, backdoor_flag=False, bddl_path=bddl_dir, seed=seed
                )
                env.reset()
                obs = None
                for _ in range(NUM_STEPS_WAIT):
                    obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
                img = get_libero_image(obs, 224)
                env.close()

                image = Image.fromarray(img).convert("RGB")
                im = tf.convert_to_tensor(np.array(image))
                dt = im.dtype
                im = tf.image.convert_image_dtype(im, tf.float32)
                im = crop_and_resize(im, 0.9, 1)
                im = tf.clip_by_value(im, 0, 1)
                im = tf.image.convert_image_dtype(im, dt, saturate=True)
                image = Image.fromarray(im.numpy()).convert("RGB")

                prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
                inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    output = vla(**inputs, output_attentions=True)

                mask = (inputs.input_ids < action_tokenizer.action_token_begin_idx)[0]
                n_layers = len(output.attentions)

                # All 32 layers again: one f_norm per layer for this sample.
                f_norm_per_layer = []
                for layer in range(n_layers):
                    _, text2patch = get_avg_patch_text_attention(
                        output.attentions, num_patches, mask, layer=layer
                    )
                    f_norm_per_layer.append(compute_fnorm(text2patch.float().cpu().numpy()))

                records.append(
                    {
                        "task_id": task_id,
                        "seed": seed,
                        "cond": cond,
                        "frame_checksum": float(img.astype(np.float64).mean()),
                        "f_norm_per_layer": f_norm_per_layer,
                    }
                )
                print(f"task={task_id} seed={seed} {cond:6s} "
                      f"L0 f={f_norm_per_layer[0]:.4f}  Llast f={f_norm_per_layer[-1]:.4f}", flush=True)

                del output
                torch.cuda.empty_cache()

    n_layers = len(records[0]["f_norm_per_layer"])
    arrays = {
        "task_id": np.array([r["task_id"] for r in records]),
        "seed": np.array([r["seed"] for r in records]),
        "cond": np.array([r["cond"] for r in records]),
        "frame_checksum": np.array([r["frame_checksum"] for r in records]),
        "f_norm": np.array([r["f_norm_per_layer"] for r in records]),  # shape (n_samples, n_layers)
    }

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(OUT_NPZ, **arrays)
    print(f"\nSaved {len(records)} samples x {n_layers} layers -> {OUT_NPZ}", flush=True)


if __name__ == "__main__":
    main()
