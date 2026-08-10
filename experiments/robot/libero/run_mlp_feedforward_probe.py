"""
experiments/robot/libero/run_mlp_feedforward_probe.py

CALIBRATION-FREE HUNT IN THE FEEDFORWARD (MLP) LAYERS.

T2IShield's F-norm feels "calibration-free" because the statistic is
SELF-NORMALIZED: it is a dimensionless number computed entirely inside one
sample, so a single universal constant can serve as the threshold. Our current
detectors (Mahalanobis drift, band-concentration z-score) all need clean
calibration data. This probe searches the MLP feedforward of all 32 LLM layers
for statistics with the same self-normalized property:

  A. SPATIAL MLP-NORM OUTLIER (per layer):
       L2 norm of each image patch's 11008-dim MLP intermediate activation
       -> top-patch / median-patch ratio, spatial Gini.
     Rationale: if the trigger hijacks a few "backdoor neurons", the trigger
     patch should be an outlier in MLP-activation space, not just attention.

  B. NEURON EXTREMITY AT ACTION TOKENS (per layer, per DoF step):
       over the 11008 neurons: max|a|/median|a|, excess kurtosis, Gini,
       fraction of neurons above 10x median.
     Rationale: backdoor-neuron literature (Fine-Pruning, ANP) says triggers
     ride on a handful of hyperactive neurons.

  C. CROSS-DoF MLP ASSIMILATION (per layer):
       mean pairwise cosine between the 7 DoF steps' neuron-activation vectors.
     Rationale: T2IShield's assimilation phenomenon, transplanted from
     attention-weight space to feedforward space and to ACTION tokens. If the
     trigger forces all 7 DoFs through the same backdoor circuit, their MLP
     activations should collapse together -- measurable inside ONE sample.

Protocol matches run_specificity_benchmark.py exactly: 5 roles (clean_cal,
clean_test, poison, decoy_ketchup, decoy_milk), disjoint seeds, all 10 tasks,
29871 append replicated. clean_cal is kept only for optional later comparison;
the analysis focuses on FIXED-THRESHOLD separation (min poison vs max benign).

Outputs: attn_maps/mlp_feedforward_probe.npz
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
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.libero.run_libero_eval_attentionmap import crop_and_resize
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer

CHECKPOINT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
OUT_NPZ = f"{REPO}/attn_maps/mlp_feedforward_probe.npz"

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
_all = [s for v in ROLE_SEEDS.values() for s in v]
assert len(_all) == len(set(_all)), "all role seeds must be disjoint"

SUITE = "libero_goal"
NUM_STEPS_WAIT = 10
DEVICE = 0
N_DOF = 7
N_LAYERS = 32

# smoke-test switches (env vars so the full launch needs no edits)
MAX_TASKS = int(os.environ.get("MLP_PROBE_MAX_TASKS", "10"))
MAX_SEEDS = int(os.environ.get("MLP_PROBE_MAX_SEEDS", "5"))


def gini(x):
    """Gini coefficient of a non-negative 1-D array (0=uniform, 1=one hot)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.shape[0]
    s = x.sum()
    if s <= 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum() / (n * s)) - (n + 1.0) / n)


def neuron_stats(v):
    """Self-normalized stats over a 1-D neuron activation vector (abs values)."""
    a = np.abs(np.asarray(v, dtype=np.float64))
    med = np.median(a)
    med = max(med, 1e-12)
    mu, sd = a.mean(), a.std()
    kurt = float(((a - mu) ** 4).mean() / max(sd ** 4, 1e-24) - 3.0)
    return (
        float(a.max() / med),          # max/median ratio
        kurt,                          # excess kurtosis
        gini(a),                       # gini
        float((a > 10.0 * med).mean()) # fraction of "massive" neurons
    )


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
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches

    ds_stats_path = os.path.join(CHECKPOINT, "dataset_statistics.json")
    if os.path.isfile(ds_stats_path):
        with open(ds_stats_path, "r") as f:
            vla.norm_stats = json.load(f)
    print(f"Model loaded. num_patches={num_patches}", flush=True)

    # ---- hooks: capture down_proj INPUT (the 11008-dim neuron activations) ----
    # prompt forward (seq>1): keep image-patch rows reduced to per-patch L2 norm
    # generation steps (seq==1): keep the full neuron vector of the new token
    capture = {"prompt_patch_norms": {}, "gen_vecs": {}}

    def make_hook(layer_idx):
        def hook(module, args):
            h = args[0]  # (B, S, 11008) bf16
            S = h.shape[1]
            if S > 1:  # prompt forward; rows 1 .. 1+num_patches are image patches
                patches = h[0, 1:1 + num_patches].float()
                capture["prompt_patch_norms"][layer_idx] = (
                    patches.norm(dim=-1).cpu().numpy())            # (256,)
            else:      # one generated (action) token
                capture["gen_vecs"].setdefault(layer_idx, []).append(
                    h[0, 0].float().cpu().numpy())                 # (11008,)
        return hook

    handles = [
        vla.language_model.model.layers[i].mlp.down_proj.register_forward_pre_hook(make_hook(i))
        for i in range(N_LAYERS)
    ]
    assert len(handles) == N_LAYERS

    def probe(img, desc):
        capture["prompt_patch_norms"].clear()
        capture["gen_vecs"].clear()
        image = Image.fromarray(img).convert("RGB")
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
        input_ids = inputs.input_ids
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            vla.generate(input_ids, pixel_values=inputs.pixel_values,
                         max_new_tokens=N_DOF, do_sample=False)

        # ---- A. spatial outlier stats over the 256 patch MLP norms ----
        spatial_ratio = np.full(N_LAYERS, np.nan)
        spatial_gini = np.full(N_LAYERS, np.nan)
        patch_norm_maps = np.zeros((N_LAYERS, num_patches), dtype=np.float32)
        for li, pn in capture["prompt_patch_norms"].items():
            patch_norm_maps[li] = pn
            med = max(float(np.median(pn)), 1e-12)
            spatial_ratio[li] = float(pn.max() / med)
            spatial_gini[li] = gini(pn)

        # ---- B. neuron extremity per layer per DoF step ----
        neuron = np.full((N_LAYERS, N_DOF, 4), np.nan)  # ratio,kurt,gini,fracmassive
        # ---- C. cross-DoF assimilation per layer ----
        dof_cos = np.full(N_LAYERS, np.nan)
        for li, vecs in capture["gen_vecs"].items():
            V = np.stack(vecs[:N_DOF])  # (7, 11008)
            for d in range(min(N_DOF, V.shape[0])):
                neuron[li, d] = neuron_stats(V[d])
            Vn = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-12, None)
            C = Vn @ Vn.T
            iu = np.triu_indices(V.shape[0], k=1)
            dof_cos[li] = float(C[iu].mean())

        torch.cuda.empty_cache()
        return spatial_ratio, spatial_gini, patch_norm_maps, neuron, dof_cos

    task_suite = benchmark.get_benchmark_dict()[SUITE]()
    records = []
    n_tasks = min(task_suite.n_tasks, MAX_TASKS)
    for task_id in range(n_tasks):
        task = task_suite.get_task(task_id)
        for role, seeds in ROLE_SEEDS.items():
            bddl_dir = BDDL[ROLE_BDDL[role]]
            for seed in seeds[:MAX_SEEDS]:
                img, desc = get_scene(task, bddl_dir, seed)
                sr, sg, pnm, neu, dc = probe(img, desc)
                records.append(dict(task_id=task_id, seed=seed, role=role,
                                    spatial_ratio=sr, spatial_gini=sg,
                                    patch_norm_maps=pnm, neuron=neu, dof_cos=dc))
        print(f"task={task_id} done ({len(records)} scenes so far)", flush=True)

    for h in handles:
        h.remove()

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(
        OUT_NPZ,
        task_id=np.array([r["task_id"] for r in records]),
        seed=np.array([r["seed"] for r in records]),
        role=np.array([r["role"] for r in records]),
        spatial_ratio=np.stack([r["spatial_ratio"] for r in records]),   # (N, 32)
        spatial_gini=np.stack([r["spatial_gini"] for r in records]),     # (N, 32)
        patch_norm_maps=np.stack([r["patch_norm_maps"] for r in records]),  # (N, 32, 256)
        neuron=np.stack([r["neuron"] for r in records]),                 # (N, 32, 7, 4)
        dof_cos=np.stack([r["dof_cos"] for r in records]),               # (N, 32)
        neuron_stat_names=np.array(["max_over_median", "kurtosis", "gini", "frac_massive"]),
    )
    print(f"\nSaved {len(records)} scenes -> {OUT_NPZ}", flush=True)

    # ---------------- quick in-run report ----------------
    roles = np.array([r["role"] for r in records])
    is_poison = roles == "poison"
    is_benign = ~is_poison  # clean_cal + clean_test + both decoys
    from sklearn.metrics import roc_auc_score

    def report(name, vals_per_layer):
        """vals_per_layer: (N, 32). Prints best layer by fixed-threshold margin."""
        best = None
        for li in range(N_LAYERS):
            v = vals_per_layer[:, li]
            if np.isnan(v).all():
                continue
            y = is_poison.astype(int)
            ok = ~np.isnan(v)
            if len(np.unique(y[ok])) < 2:
                continue
            auc = roc_auc_score(y[ok], v[ok])
            auc = max(auc, 1 - auc)
            # fixed-threshold margin: does ANY single constant separate poison
            # from ALL benign (clean + decoys)?
            lo_p, hi_p = np.nanmin(v[is_poison]), np.nanmax(v[is_poison])
            lo_b, hi_b = np.nanmin(v[is_benign]), np.nanmax(v[is_benign])
            sep_up = lo_p > hi_b     # poison strictly above all benign
            sep_dn = hi_p < lo_b     # poison strictly below all benign
            margin = max(lo_p - hi_b, lo_b - hi_p)
            if best is None or auc > best[1]:
                best = (li, auc, sep_up or sep_dn, margin,
                        (lo_p, hi_p), (lo_b, hi_b))
        if best:
            li, auc, sep, margin, (lp, hp), (lb, hb) = best
            print(f"  {name:<26} best L{li:>2}: AUROC={auc:.3f} "
                  f"fixed-thresh-separable={'YES' if sep else 'no'} margin={margin:+.4f} "
                  f"poison=[{lp:.3f},{hp:.3f}] benign=[{lb:.3f},{hb:.3f}]")

    print("\n=== poison vs ALL benign (clean+decoys), per statistic, best layer ===")
    report("A spatial max/med ratio", np.stack([r["spatial_ratio"] for r in records]))
    report("A spatial gini", np.stack([r["spatial_gini"] for r in records]))
    neu = np.stack([r["neuron"] for r in records])  # (N,32,7,4)
    for si, sn in enumerate(["max_over_median", "kurtosis", "gini", "frac_massive"]):
        report(f"B neuron {sn} (DoF-avg)", np.nanmean(neu[:, :, :, si], axis=2))
    report("C cross-DoF cosine", np.stack([r["dof_cos"] for r in records]))


if __name__ == "__main__":
    main()
