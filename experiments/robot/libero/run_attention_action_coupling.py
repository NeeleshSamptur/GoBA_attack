"""
experiments/robot/libero/run_attention_action_coupling.py

NOVELTY EXPERIMENT: does the backdoor's ATTENTION corruption line up, per action
dimension, with its ACTION corruption?

Prior VLA backdoor defenses all collapse the model's internal state into ONE
number before deciding (T2IShield-style dispersion; TrustVLA's Dirichlet
"mechanism score"; Bera's token reconstruction). None of them decompose the
signature by action dimension -- confirmed by reading them. But a VLA's output
is not a class label, it is a 7-DoF action, and an attacker's target behavior
("grasp the trigger object") lives in specific dimensions of that action. So
the question nobody has asked is whether the attention signature is
ACTION-STRUCTURED: is the DoF whose attention is most disturbed also the DoF
whose commanded value is most disturbed?

For every scene we capture, from ONE generate() call:
  * per-DoF attention (last layer) over image patches / text tokens
      -> entropy per DoF per group      (the "attention corruption" axis)
  * the 7 decoded, un-normalized action values
      -> one value per DoF              (the "action corruption" axis)

Then, per DoF, we compute AUROC(poison vs clean) on BOTH axes independently and
ask whether they agree across the 7 DoFs. Agreement would mean the attention
signature is not a diffuse anomaly but a readout of which part of the robot's
behavior the attacker actually seized -- i.e. a detector that also tells you
WHAT the robot is about to do wrong, not merely that something is wrong.

Action decode replicates OpenVLAForActionPrediction.predict_action verbatim
(vocab_size - token_id -> bin index -> bin center -> q01/q99 unnormalize),
INCLUDING its 29871 "empty token" append, which the other attention scripts in
this repo omit; a diagnostic below reports whether that append actually fires,
since if it does those scripts measured attention on a different input than the
one the policy really acts from.

Disjoint clean/poison seeds + all 10 tasks, matching
run_attention_assimilation_detector.py.

Outputs: attn_maps/attention_action_coupling.npz + printed per-DoF tables.
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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.libero.run_libero_eval_attentionmap import crop_and_resize
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer

CHECKPOINT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
OUT_NPZ = f"{REPO}/attn_maps/attention_action_coupling.npz"

CONDITIONS = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
}
SUITE = "libero_goal"
CLEAN_SEEDS = [7, 42, 1234, 2026, 31337, 5, 99, 777, 20260803, 424242]
POISON_SEEDS = [11, 43, 1337, 2027, 31338, 6, 100, 778, 20260804, 424243]
assert not set(CLEAN_SEEDS) & set(POISON_SEEDS), "clean/poison seed sets must be disjoint"

NUM_STEPS_WAIT = 10
DEVICE = 0
N_DOF = 7
DOF_NAMES = ["dx", "dy", "dz", "roll", "pitch", "yaw", "gripper"]
GROUPS = ["image", "text"]


def _entropy(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = p / max(p.sum(), eps)
    n = p.shape[0]
    if n <= 1:
        return float("nan")
    return float(-np.sum(p * np.log(p + eps)) / np.log(n))


def get_scene_image(task, cond_dir, seed):
    env, desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False, bddl_path=cond_dir, seed=seed)
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
    return np.array(Image.fromarray(im.numpy()).convert("RGB")), desc


def directed_auroc(vals, labels):
    """Returns (magnitude, direction). Magnitude always >= 0.5."""
    vals = np.asarray(vals, dtype=np.float64)
    ok = ~np.isnan(vals)
    raw = roc_auc_score(labels[ok], vals[ok])
    return (raw, "up") if raw >= 0.5 else (1.0 - raw, "down")


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

    # get_vla() normally merges the fine-tuned checkpoint's dataset_statistics.json into
    # norm_stats; we load the model directly (get_vla forces flash_attention_2, which
    # silently returns None for output_attentions), so replicate that merge by hand.
    ds_stats_path = os.path.join(CHECKPOINT, "dataset_statistics.json")
    if os.path.isfile(ds_stats_path):
        import json
        with open(ds_stats_path, "r") as f:
            vla.norm_stats = json.load(f)

    unnorm_key = SUITE if SUITE in vla.norm_stats else f"{SUITE}_no_noops"
    stats = vla.get_action_stats(unnorm_key)
    a_low, a_high = np.array(stats["q01"]), np.array(stats["q99"])
    a_mask = np.array(stats.get("mask", np.ones_like(a_low, dtype=bool)))
    print(f"Model loaded. num_patches={num_patches} unnorm_key={unnorm_key}", flush=True)

    task_suite = benchmark.get_benchmark_dict()[SUITE]()
    append_fired = [0, 0]  # [n_appended, n_total]

    def probe(img, desc):
        image = Image.fromarray(img).convert("RGB")
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

        # Match predict_action's 29871 handling EXACTLY (the other attention scripts skip this).
        input_ids = inputs.input_ids
        append_fired[1] += 1
        if not torch.all(input_ids[:, -1] == 29871):
            append_fired[0] += 1
            input_ids = torch.cat(
                (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1
            )

        mask = (input_ids < action_tokenizer.action_token_begin_idx)[0]
        tmask = mask[1:].cpu().numpy().astype(bool)
        text_col_start = 1 + num_patches
        text_col_end = text_col_start + tmask.shape[0]

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = vla.generate(
                input_ids, pixel_values=inputs.pixel_values, max_new_tokens=N_DOF,
                output_attentions=True, return_dict_in_generate=True, do_sample=False,
            )

        layer = len(gen.attentions[0]) - 1  # last layer
        ent = {g: np.full(N_DOF, np.nan) for g in GROUPS}
        img_maps = np.zeros((N_DOF, num_patches), dtype=np.float32)  # kept for localization analysis
        for dof_idx, step_attn in enumerate(gen.attentions):
            la = step_attn[layer][0].float().mean(dim=0)          # avg over heads
            row = (la[-1] if la.dim() == 2 else la[0]).cpu().numpy()
            img_part = row[1: 1 + num_patches]
            img_maps[dof_idx] = img_part
            ent["image"][dof_idx] = _entropy(img_part)
            ent["text"][dof_idx] = _entropy(row[text_col_start:text_col_end][tmask])

        # Decode the 7 action-bin tokens -> continuous action (predict_action logic, verbatim).
        tok = gen.sequences[0, -N_DOF:].cpu().numpy()
        disc = vla.vocab_size - tok
        disc = np.clip(disc - 1, a_min=0, a_max=vla.bin_centers.shape[0] - 1)
        normed = vla.bin_centers[disc]
        action = np.where(a_mask, 0.5 * (normed + 1) * (a_high - a_low) + a_low, normed)

        del gen
        torch.cuda.empty_cache()
        return ent, action.astype(np.float64), img_maps

    records = []
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        for cond, bddl_dir in CONDITIONS.items():
            for seed in (CLEAN_SEEDS if cond == "clean" else POISON_SEEDS):
                img, desc = get_scene_image(task, bddl_dir, seed)
                ent, action, img_maps = probe(img, desc)
                records.append({"task_id": task_id, "seed": seed, "cond": cond,
                                "ent": ent, "action": action, "img_maps": img_maps})
        print(f"task={task_id} done ({len(records)} samples so far)", flush=True)

    labels = np.array([1 if r["cond"] == "poison" else 0 for r in records])
    actions = np.stack([r["action"] for r in records])              # (N, 7)
    ent_img = np.stack([r["ent"]["image"] for r in records])         # (N, 7)
    ent_txt = np.stack([r["ent"]["text"] for r in records])          # (N, 7)

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(OUT_NPZ,
             task_id=np.array([r["task_id"] for r in records]),
             seed=np.array([r["seed"] for r in records]),
             cond=np.array([r["cond"] for r in records]),
             actions=actions, entropy_image=ent_img, entropy_text=ent_txt,
             img_maps=np.stack([r["img_maps"] for r in records]),  # (N, 7, 256)
             dof_names=np.array(DOF_NAMES))
    print(f"\nSaved {len(records)} samples -> {OUT_NPZ}")
    print(f"29871 append fired on {append_fired[0]}/{append_fired[1]} samples "
          f"({'MATTERS -- other scripts omit this' if append_fired[0] else 'no-op here'})")

    print("\n" + "=" * 88)
    print("PER-DoF: does ATTENTION corruption track ACTION corruption? (disjoint seeds, last layer)")
    print("All AUROCs direction-corrected (magnitude >= 0.5); 'dir' says which way the trigger moved it.")
    print("=" * 88)
    hdr = (f"{'DoF':<8} | {'ACTION auroc':>13} {'dir':>5} | {'attn-IMG auroc':>15} {'dir':>5} "
           f"| {'attn-TXT auroc':>15} {'dir':>5}")
    print(hdr); print("-" * len(hdr))
    act_a, img_a, txt_a = [], [], []
    for i, name in enumerate(DOF_NAMES):
        aa, ad = directed_auroc(actions[:, i], labels)
        ia, idr = directed_auroc(ent_img[:, i], labels)
        ta, tdr = directed_auroc(ent_txt[:, i], labels)
        act_a.append(aa); img_a.append(ia); txt_a.append(ta)
        print(f"{name:<8} | {aa:>13.4f} {ad:>5} | {ia:>15.4f} {idr:>5} | {ta:>15.4f} {tdr:>5}")

    act_a, img_a, txt_a = np.array(act_a), np.array(img_a), np.array(txt_a)
    print("\n--- COUPLING across the 7 DoFs (does attention AUROC predict action AUROC?) ---")
    for nm, arr in (("attn-IMAGE", img_a), ("attn-TEXT", txt_a)):
        pr, pp = pearsonr(arr, act_a)
        sr, sp = spearmanr(arr, act_a)
        print(f"  {nm:<11} vs ACTION:  pearson r={pr:+.3f} (p={pp:.3f})   spearman rho={sr:+.3f} (p={sp:.3f})")
    print("\n  Ranking by ACTION corruption:   " + ", ".join(
        f"{DOF_NAMES[i]}({act_a[i]:.2f})" for i in np.argsort(-act_a)))
    print("  Ranking by ATTENTION (image):   " + ", ".join(
        f"{DOF_NAMES[i]}({img_a[i]:.2f})" for i in np.argsort(-img_a)))


if __name__ == "__main__":
    main()
