"""
experiments/robot/libero/run_flow_entropy_detector.py

ACTION-TO-PERCEPT FLOW ENTROPY: a backdoor statistic that is NOT T2IShield.

Every attention-based backdoor detector (T2IShield FTT/CDA, AttenTD, AHTD,
Bera, TrustVLA) reads SINGLE-LAYER RAW ATTENTION WEIGHTS. The interpretability
literature identified two flaws in raw weights years ago, and neither fix has
ever been used for backdoor detection:
  * across layers, token identities mix, so layer-l weights do not reflect
    flow to the INPUT -- fixed by attention ROLLOUT (Abnar & Zuidema, ACL'20):
    R = prod_l (0.5*A_l + 0.5*I), row-stochastic;
  * weights ignore the magnitude of what is mixed -- fixed by VALUE-WEIGHTING
    (Kobayashi et al., EMNLP'20): A_hat[i,j] ~ mean_h alpha_h[i,j]*||v_h(j)||.

This detector composes both fixes and moves the query side to ACTION tokens:
for each generated action token, compute its rolled-out (value-weighted) flow
distribution over the 256 image patches, and score its normalized Shannon
ENTROPY. Hypothesis: the trigger funnels the accumulated action->percept flow
through its own patch, collapsing entropy; benign novel objects attract raw
glances but not accumulated flow.

Design properties vs T2IShield: activations not weights; multi-layer flow not
one layer (all 32 layers enter BY CONSTRUCTION -- no layer selection at all);
entropy not F-norm; action queries not text. Baselines computed from the SAME
forward pass: single-layer last-layer raw-weight entropy (AttenTD/Bera-style)
and T2IShield F-norm, so the comparison is not confounded.

Protocol: identical to run_specificity_benchmark.py (5 roles, disjoint seeds,
10 tasks, 29871-corrected). Outputs: attn_maps/flow_entropy.npz
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
from libero.libero import benchmark
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.libero.run_libero_eval_attentionmap import crop_and_resize
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

CHECKPOINT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
OUT_NPZ = f"{REPO}/attn_maps/flow_entropy.npz"

BDDL = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
    "decoy_ketchup": f"{REPO}/BadLIBERO/libero/libero/bddl_files-decoy_eval",
    "decoy_milk": f"{REPO}/BadLIBERO/libero/libero/bddl_files-decoy2_eval",
}
ROLE_SEEDS = {
    "clean_cal":     [7, 42, 1234, 2026, 31337],
    "clean_test":    [5, 99, 777, 20260803, 424242],
    "poison":        [11, 43, 1337, 2027, 31338],
    "decoy_ketchup": [6, 100, 778, 20260804, 424243],
    "decoy_milk":    [8, 101, 779, 20260805, 424244],
}
ROLE_BDDL = {
    "clean_cal": "clean", "clean_test": "clean", "poison": "poison",
    "decoy_ketchup": "decoy_ketchup", "decoy_milk": "decoy_milk",
}
SUITE = "libero_goal"
NUM_STEPS_WAIT = 10
DEVICE = 0
N_DOF = 7
N_GEN = 6          # action tokens that get a forward pass (dx..yaw)
N_LAYERS = 32

MAX_TASKS = int(os.environ.get("FLOW_MAX_TASKS", "10"))
MAX_SEEDS = int(os.environ.get("FLOW_MAX_SEEDS", "5"))


def norm_entropy(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    if s <= eps or p.shape[0] <= 1:
        return float("nan")
    p = p / s
    return float(-np.sum(p * np.log(p + eps)) / np.log(p.shape[0]))


def get_scene(task, bddl_dir, seed):
    env, desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False,
                               bddl_path=bddl_dir, seed=seed)
    env.reset()
    obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
    img = get_libero_image(obs, 224)
    env.close()
    im = tf.convert_to_tensor(np.array(Image.fromarray(img).convert("RGB")))
    dt = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, dt, saturate=True)
    return np.array(Image.fromarray(im.numpy()).convert("RGB")), desc


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
    vla.norm_stats = json.load(open(os.path.join(CHECKPOINT, "dataset_statistics.json")))
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches
    n_heads = vla.language_model.config.num_attention_heads
    head_dim = vla.language_model.config.hidden_size // n_heads
    print(f"Model loaded. num_patches={num_patches} heads={n_heads}", flush=True)

    # hooks: per-layer, per-position, per-head VALUE norms ||v_h(j)||
    vnorms = {}  # layer -> list of (positions, heads) arrays, in call order

    def make_vhook(li):
        def hook(module, args, output):
            # output: (B, S, n_heads*head_dim)
            v = output[0].float().view(-1, n_heads, head_dim)   # (S, H, D)
            vnorms.setdefault(li, []).append(v.norm(dim=-1))    # (S, H) on GPU
        return hook

    handles = [
        vla.language_model.model.layers[i].self_attn.v_proj.register_forward_hook(make_vhook(i))
        for i in range(N_LAYERS)
    ]

    def probe(img, desc):
        """Returns per-scene statistics from one generate() call."""
        vnorms.clear()
        image = Image.fromarray(img).convert("RGB")
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
        input_ids = inputs.input_ids
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = vla.generate(input_ids, pixel_values=inputs.pixel_values,
                               max_new_tokens=N_DOF, output_attentions=True,
                               return_dict_in_generate=True, do_sample=False)

        S = gen.attentions[0][0].shape[-1]        # prompt length (LLM side, incl patches)
        T = S + N_GEN                             # final processed length (gripper token unprocessed)
        img_cols = slice(1, 1 + num_patches)

        # per-layer value norms for all T positions: (T, H)
        vn = {}
        for li, chunks in vnorms.items():
            vn[li] = torch.cat(chunks, dim=0)[:T]  # prompt (S,H) + 6 steps of (1,H)

        # build full (T,T) attention matrices per layer, weight-based and value-weighted
        R_w = torch.eye(T, device=DEVICE)
        R_v = torch.eye(T, device=DEVICE)
        # cache last-layer raw rows for the single-layer baselines
        last_raw_rows = None
        eye = torch.eye(T, device=DEVICE)
        for li in range(N_LAYERS):
            A_h = torch.zeros(n_heads, T, T, device=DEVICE)
            # prompt block
            A_h[:, :S, :S] = gen.attentions[0][li][0].float()
            # step rows: step k (k=1..6) processed position S+k-1, row length S+k
            for k in range(1, N_GEN + 1):
                row = gen.attentions[k][li][0].float()  # (H, 1, S+k)
                A_h[:, S + k - 1, :S + k] = row[:, 0, :]

            # weight-based: head-average
            A_w = A_h.mean(dim=0)
            # value-weighted: mean_h alpha_h[i,j] * ||v_h(j)||, then row-normalize
            A_v = (A_h * vn[li].t().unsqueeze(1)).mean(dim=0)   # (T,T); vn.t(): (H,T)
            A_v = A_v / A_v.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            if li == N_LAYERS - 1:
                last_raw_rows = A_w.clone()

            # residual-corrected rollout update
            R_w = (0.5 * A_w + 0.5 * eye) @ R_w
            R_v = (0.5 * A_v + 0.5 * eye) @ R_v

        # per-action-token flow over image patches (positions S..S+5 = dx..yaw)
        ent_roll_w = np.full(N_GEN, np.nan)
        ent_roll_v = np.full(N_GEN, np.nan)
        conc_roll_v = np.full(N_GEN, np.nan)
        ent_last = np.full(N_GEN, np.nan)      # single-layer baseline (AttenTD/Bera style)
        flow_maps = np.zeros((N_GEN, num_patches), dtype=np.float32)
        for i in range(N_GEN):
            r = S + i
            fw = R_w[r, img_cols].cpu().numpy()
            fv = R_v[r, img_cols].cpu().numpy()
            flow_maps[i] = fv
            ent_roll_w[i] = norm_entropy(fw)
            ent_roll_v[i] = norm_entropy(fv)
            conc_roll_v[i] = float(fv.max() / max(fv.sum(), 1e-12))
            ent_last[i] = norm_entropy(last_raw_rows[r, img_cols].cpu().numpy())

        # T2IShield F-norm baseline: text->patch rows of the last layer
        t0 = 1 + num_patches
        t2p = last_raw_rows[t0:S, img_cols].cpu().numpy()
        p = t2p / np.clip(t2p.sum(axis=1, keepdims=True), 1e-8, None)
        fnorm = float(np.linalg.norm(p - p.mean(axis=0, keepdims=True), axis=1).mean())

        del gen
        torch.cuda.empty_cache()
        return (ent_roll_w.mean(), ent_roll_v.mean(), np.nanmean(conc_roll_v),
                ent_last.mean(), fnorm, flow_maps)

    task_suite = benchmark.get_benchmark_dict()[SUITE]()
    records = []
    n_tasks = min(task_suite.n_tasks, MAX_TASKS)
    for task_id in range(n_tasks):
        task = task_suite.get_task(task_id)
        for role, seeds in ROLE_SEEDS.items():
            bddl_dir = BDDL[ROLE_BDDL[role]]
            for seed in seeds[:MAX_SEEDS]:
                img, desc = get_scene(task, bddl_dir, seed)
                erw, erv, crv, el, fn, fm = probe(img, desc)
                records.append(dict(task_id=task_id, seed=seed, role=role,
                                    ent_roll_w=erw, ent_roll_v=erv, conc_roll_v=crv,
                                    ent_last=el, fnorm=fn, flow_maps=fm))
        print(f"task={task_id} done ({len(records)} scenes)", flush=True)

    for h in handles:
        h.remove()

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(OUT_NPZ,
             task_id=np.array([r["task_id"] for r in records]),
             seed=np.array([r["seed"] for r in records]),
             role=np.array([r["role"] for r in records]),
             ent_roll_w=np.array([r["ent_roll_w"] for r in records]),
             ent_roll_v=np.array([r["ent_roll_v"] for r in records]),
             conc_roll_v=np.array([r["conc_roll_v"] for r in records]),
             ent_last=np.array([r["ent_last"] for r in records]),
             fnorm=np.array([r["fnorm"] for r in records]),
             flow_maps=np.stack([r["flow_maps"] for r in records]))
    print(f"\nSaved {len(records)} scenes -> {OUT_NPZ}", flush=True)

    # ---------------- report ----------------
    role = np.array([r["role"] for r in records])
    task = np.array([r["task_id"] for r in records])
    is_p = role == "poison"
    is_c = np.isin(role, ["clean_cal", "clean_test"])
    is_k, is_m = role == "decoy_ketchup", role == "decoy_milk"
    val, test = task <= 4, task >= 5

    def dirauc(v, p, ne):
        ok = ~np.isnan(v)
        y = np.r_[np.ones((p & ok).sum()), np.zeros((ne & ok).sum())]
        s = np.r_[v[p & ok], v[ne & ok]]
        r = roc_auc_score(y, s)
        return r if r >= 0.5 else 1 - r

    print("\n=== detection & specificity (all scenes) ===")
    print(f"{'statistic':<28} {'poisonAUC':>9} {'ketchAUC':>9} {'milkAUC':>8} {'gap':>7}")
    for name, key in [("FLOW entropy (value-wtd)", "ent_roll_v"),
                      ("FLOW entropy (weight)", "ent_roll_w"),
                      ("FLOW concentration (vw)", "conc_roll_v"),
                      ("last-layer entropy (base)", "ent_last"),
                      ("T2IShield F-norm (base)", "fnorm")]:
        v = np.array([r[key] for r in records])
        ap = dirauc(v, is_p, is_c)
        ak = dirauc(v, is_k, is_c)
        am = dirauc(v, is_m, is_c)
        print(f"{name:<28} {ap:>9.4f} {ak:>9.4f} {am:>8.4f} {ap - max(ak, am):>+7.4f}")

    print("\n=== fixed-threshold check on FLOW entropy (value-wtd) ===")
    v = np.array([r["ent_roll_v"] for r in records])
    for nm, msk in [("poison", is_p), ("clean", is_c), ("ketchup", is_k), ("milk", is_m)]:
        print(f"  {nm:<8} [{np.nanmin(v[msk]):.4f}, {np.nanmax(v[msk]):.4f}]  mean={np.nanmean(v[msk]):.4f}")
    is_b = ~is_p
    m = max(np.nanmin(v[is_p]) - np.nanmax(v[is_b]), np.nanmin(v[is_b]) - np.nanmax(v[is_p]))
    print(f"  global fixed margin = {m:+.4f}")
    pv, bv = v[val & is_p], v[val & is_b]
    up = np.nanmedian(pv) > np.nanmedian(bv)
    thr = (np.nanmin(pv) + np.nanmax(bv)) / 2 if up else (np.nanmax(pv) + np.nanmin(bv)) / 2
    fire = (lambda x: x > thr) if up else (lambda x: x < thr)
    print(f"  VAL-frozen thr={thr:.4f} dir={'up' if up else 'down'} -> TEST: "
          f"poison {100*np.nanmean(fire(v[test&is_p])):.0f}%  clean {100*np.nanmean(fire(v[test&is_c])):.0f}%  "
          f"ketchup {100*np.nanmean(fire(v[test&is_k])):.0f}%  milk {100*np.nanmean(fire(v[test&is_m])):.0f}%")


if __name__ == "__main__":
    main()
