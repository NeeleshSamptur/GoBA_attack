"""
experiments/robot/libero/run_norm_based_action_attribution.py

NORM-BASED (VALUE-WEIGHTED) ATTENTION FOR VLA BACKDOOR DIAGNOSIS
-- and a causal attention->action attribution no prior defense can compute.

Motivation. Every attention-based backdoor defense to date is WEIGHT-based:
T2IShield's FTT takes the F-norm of raw cross-attention maps; AttenTD / AHTD
(Trojaned BERT/ViT) monitor raw-weight drift; Bera's token saliency is a plain
column-mean of raw attention weights (their Eq. 7); TrustVLA scores token-level
evidence. But Kobayashi et al. (EMNLP 2020) showed raw attention weights
mismeasure information flow: what actually enters a token's residual stream
from source j is  || alpha_ij * f(x_j) ||  with  f(x_j) = W_O^h W_V^h x_j,
and large weights are routinely cancelled by small value norms. Norm-based
analysis has never been used for backdoor detection in any modality.

A VLA lets us push one step further than Kobayashi: the last-layer attention
output at an action-token position feeds LINEARLY (o_proj -> RMSNorm gain ->
lm_head row of the chosen action-bin token) into the action logit. So each
image patch's contribution to the COMMANDED ACTION is a computable scalar:

    contrib_k(j) = sum_h alpha_h[q_k, j] * ( w_eff . f_h(x_j) ) / rms(h_pre[q_k])
    w_eff = lm_head[token_k] * rmsnorm_gain          (direct logit attribution)

This is a causal (linear-path) attention->action link, not a correlation.

From ONE teacher-forced forward pass per scene (after generating the 7 action
tokens with the predict_action-faithful 29871 handling) we compute, per layer:
  * weight-based maps      alpha (head-avg)                 [T2IShield lineage]
  * norm-weighted maps     sum_h alpha_h * ||W_O^h v_h||    [ours]
for both TEXT-token query rows (the existing detector's construction) and
ACTION-token query rows (7 per scene, one per DoF), plus last-layer
action-logit attribution maps. The same FTT-style dispersion statistic
(mean_row ||p_row - p_mean||) is applied identically to weight and norm maps,
so any performance difference is attributable to value-weighting alone.

Protocol: the 250-scene / 5-role specificity benchmark (clean_cal, clean_test,
poison, decoy_ketchup, decoy_milk; all seeds disjoint), so every statistic is
scored on BOTH axes: detection AUROC (poison vs clean) and specificity gap
(poison AUROC minus benign-decoy AUROC). Localization hit-rates in the trigger
footprint are also compared weight vs norm vs logit-attribution.

Outputs: attn_maps/norm_based_attribution.npz + printed comparison tables.
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
from prismatic.vla.action_tokenizer import ActionTokenizer

CHECKPOINT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
OUT_NPZ = f"{REPO}/attn_maps/norm_based_attribution.npz"

BDDL = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
    "decoy_ketchup": f"{REPO}/BadLIBERO/libero/libero/bddl_files-decoy_eval",
    "decoy_milk": f"{REPO}/BadLIBERO/libero/libero/bddl_files-decoy2_eval",
}
# Same role->seed assignment as run_specificity_benchmark.py (scene-for-scene matched).
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
DOF_NAMES = ["dx", "dy", "dz", "roll", "pitch", "yaw", "gripper"]
MAP_LAYERS = [0, 8, 16, 24, 27, 31]   # layers whose full action maps are stored
GRID = 16
TRIG_ROWS, TRIG_COLS = slice(11, 15), slice(0, 4)   # eval-only ground truth box


def fnorm_stat(maps):
    """T2IShield-FTT-style dispersion, applied identically to weight and norm maps."""
    p = maps / np.clip(maps.sum(axis=1, keepdims=True), 1e-12, None)
    return float(np.linalg.norm(p - p.mean(axis=0)[None, :], axis=1).mean())


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


def directed(vals, pos, neg):
    ok = ~np.isnan(vals)
    y = np.concatenate([np.ones((pos & ok).sum()), np.zeros((neg & ok).sum())])
    s = np.concatenate([vals[pos & ok], vals[neg & ok]])
    raw = roc_auc_score(y, s)
    return raw if raw >= 0.5 else 1.0 - raw


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
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches

    llm = vla.language_model
    layers = llm.model.layers
    n_layers = len(layers)
    attn0 = layers[0].self_attn
    n_heads, head_dim = attn0.num_heads, attn0.head_dim
    assert getattr(attn0, "num_key_value_heads", n_heads) == n_heads, "expected MHA (no GQA)"
    hidden = n_heads * head_dim
    print(f"Model loaded. layers={n_layers} heads={n_heads} head_dim={head_dim}", flush=True)

    # Hooks: per-layer value states + pre-final-norm hidden states.
    v_cache, prenorm = {}, {}
    for l, lyr in enumerate(layers):
        lyr.self_attn.v_proj.register_forward_hook(
            lambda _m, _i, out, l=l: v_cache.__setitem__(l, out))
    llm.model.norm.register_forward_pre_hook(
        lambda _m, inp: prenorm.__setitem__("h", inp[0]))

    # Per-head output transforms: o = sum_h Wo_h @ v_h with Wo_h = W_O[:, h*d:(h+1)*d].
    Wo = {l: layers[l].self_attn.o_proj.weight.detach().view(hidden, n_heads, head_dim)
          for l in range(n_layers)}
    lm_head_w = llm.lm_head.weight.detach()
    norm_gain = llm.model.norm.weight.detach().float()
    # Action-bin token ids (decode maps token -> bin via vocab_size - token - 1 in [0, 254]).
    bin_ids = torch.arange(vla.vocab_size - 255, vla.vocab_size, device=lm_head_w.device)
    bin_mean_w = lm_head_w[bin_ids].float().mean(dim=0)      # mean unembedding over bins

    task_suite = benchmark.get_benchmark_dict()[SUITE]()

    def probe(img, desc):
        image = Image.fromarray(img).convert("RGB")
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
        input_ids = inputs.input_ids
        if not torch.all(input_ids[:, -1] == 29871):   # predict_action-faithful
            input_ids = torch.cat(
                (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = vla.generate(input_ids, pixel_values=inputs.pixel_values,
                               max_new_tokens=N_DOF, do_sample=False)
        action_ids = gen[0, -N_DOF:]

        # One teacher-forced pass over prompt + generated action tokens.
        full_ids = torch.cat((input_ids, action_ids.unsqueeze(0)), dim=1)
        v_cache.clear()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = vla(input_ids=full_ids, pixel_values=inputs.pixel_values,
                      output_attentions=True)

        Lp = input_ids.shape[1]
        # LLM sequence: [BOS, 256 patches, text tokens 1..Lp-1, 7 action tokens]
        patch_cols = slice(1, 1 + num_patches)
        q_rows = [num_patches + Lp - 1 + k for k in range(N_DOF)]  # row producing token k
        tmask = (input_ids < action_tokenizer.action_token_begin_idx)[0][1:].cpu().numpy().astype(bool)
        text_rows = (num_patches + 1 + np.nonzero(tmask)[0]).tolist()

        N = out.attentions[0].shape[-1]
        assert N == num_patches + Lp + N_DOF, f"seq mismatch {N}"

        h_pre = prenorm["h"][0].float()                       # (N, hidden), pre-final-norm
        rms = h_pre.pow(2).mean(dim=-1).add(1e-6).sqrt()      # (N,)

        rec = {"fw_txt": np.zeros(n_layers, np.float32), "fn_txt": np.zeros(n_layers, np.float32),
               "fw_act": np.zeros(n_layers, np.float32), "fn_act": np.zeros(n_layers, np.float32),
               "maps_w": {}, "maps_n": {}}
        for l in range(n_layers):
            A = out.attentions[l][0].float()                  # (heads, N, N)
            vh = v_cache[l][0].float().view(N, n_heads, head_dim)
            f = torch.einsum("nhd,ohd->nho", vh, Wo[l].float())   # (N, heads, hidden)
            f_norms = f.norm(dim=-1)                          # (N, heads): ||W_O^h v_h(j)||

            A_act = A[:, q_rows, :]                           # (heads, 7, N)
            A_txt = A[:, text_rows, :]
            w_act = A_act.mean(dim=0)[:, patch_cols]          # weight-based, head-avg
            n_act = torch.einsum("hqn,nh->qn", A_act, f_norms)[:, patch_cols]
            w_txt = A_txt.mean(dim=0)[:, patch_cols]
            n_txt = torch.einsum("hqn,nh->qn", A_txt, f_norms)[:, patch_cols]

            rec["fw_act"][l] = fnorm_stat(w_act.cpu().numpy())
            rec["fn_act"][l] = fnorm_stat(n_act.cpu().numpy())
            rec["fw_txt"][l] = fnorm_stat(w_txt.cpu().numpy())
            rec["fn_txt"][l] = fnorm_stat(n_txt.cpu().numpy())
            if l in MAP_LAYERS:
                rec["maps_w"][l] = w_act.cpu().numpy().astype(np.float32)
                rec["maps_n"][l] = n_act.cpu().numpy().astype(np.float32)

            if l == n_layers - 1:
                # Direct logit attribution: contribution of each patch to the chosen
                # action-bin logit CONTRASTED against the mean action-bin logit
                # ("why this bin rather than another"), through o_proj -> RMSNorm
                # gain -> lm_head (linear path; rms treated as constant).
                attrib = np.zeros((N_DOF, num_patches), np.float32)
                for k in range(N_DOF):
                    w_eff = (lm_head_w[action_ids[k]].float() - bin_mean_w) * norm_gain
                    proj_f = f @ w_eff                                            # (N, heads)
                    c = (A[:, q_rows[k], :].T * proj_f).sum(dim=1) / rms[q_rows[k]]
                    attrib[k] = c[patch_cols].cpu().numpy()
                rec["attrib"] = attrib
            del A, vh, f, f_norms
        del out
        torch.cuda.empty_cache()
        return rec

    recs = []
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        for role, seeds in ROLE_SEEDS.items():
            bddl_dir = BDDL[ROLE_BDDL[role]]
            for seed in seeds:
                img, desc = get_scene(task, bddl_dir, seed)
                rec = probe(img, desc)
                rec.update(task_id=task_id, seed=seed, role=role)
                recs.append(rec)
        print(f"task={task_id} done ({len(recs)} scenes)", flush=True)

    roles = np.array([r["role"] for r in recs])
    save = {
        "role": roles,
        "task_id": np.array([r["task_id"] for r in recs]),
        "seed": np.array([r["seed"] for r in recs]),
        "fw_txt": np.stack([r["fw_txt"] for r in recs]),
        "fn_txt": np.stack([r["fn_txt"] for r in recs]),
        "fw_act": np.stack([r["fw_act"] for r in recs]),
        "fn_act": np.stack([r["fn_act"] for r in recs]),
        "attrib": np.stack([r["attrib"] for r in recs]),
        "map_layers": np.array(MAP_LAYERS),
    }
    for l in MAP_LAYERS:
        save[f"maps_w_L{l}"] = np.stack([r["maps_w"][l] for r in recs])
        save[f"maps_n_L{l}"] = np.stack([r["maps_n"][l] for r in recs])
    np.savez(OUT_NPZ, **save)
    print(f"\nSaved {len(recs)} scenes -> {OUT_NPZ}")

    # ================= ANALYSIS =================
    is_p = roles == "poison"
    ct = roles == "clean_test"
    is_k, is_m = roles == "decoy_ketchup", roles == "decoy_milk"

    def report_row(name, vals):
        a_p = directed(vals, is_p, ct)
        a_k = directed(vals, is_k, ct)
        a_m = directed(vals, is_m, ct)
        print(f"{name:<26} | {a_p:>8.4f} | {a_k:>8.4f} | {a_m:>7.4f} | {a_p - max(a_k, a_m):>+10.4f}")

    print("\n" + "=" * 78)
    print("WEIGHT-BASED vs NORM-BASED F-NORM STATISTIC (same formula, same rows)")
    print("=" * 78)
    print(f"{'statistic':<26} | {'poison':>8} | {'ketchup':>8} | {'milk':>7} | {'spec. gap':>10}")
    print("-" * 72)
    fw_txt, fn_txt = save["fw_txt"], save["fn_txt"]
    fw_act, fn_act = save["fw_act"], save["fn_act"]
    for l in [16, 24, 27, 31]:
        report_row(f"L{l} text  WEIGHT (T2IS)", fw_txt[:, l])
        report_row(f"L{l} text  NORM  (ours)", fn_txt[:, l])
        report_row(f"L{l} action WEIGHT", fw_act[:, l])
        report_row(f"L{l} action NORM  (ours)", fn_act[:, l])
        print("-" * 72)

    print("\nPer-layer detection AUROC (poison vs clean), text rows: weight vs norm")
    print(f"{'layer':>5} {'weight':>8} {'norm':>8}   {'layer':>5} {'weight':>8} {'norm':>8}")
    n_layers_ = fw_txt.shape[1]
    half = (n_layers_ + 1) // 2
    for i in range(half):
        row = ""
        for l in (i, i + half):
            if l < n_layers_:
                aw = directed(fw_txt[:, l], is_p, ct)
                an = directed(fn_txt[:, l], is_p, ct)
                row += f"{l:>5} {aw:>8.3f} {an:>8.3f}   "
        print(row)

    # ---- localization: weight vs norm vs logit-attribution maps ----
    box = np.zeros((GRID, GRID), bool)
    box[TRIG_ROWS, TRIG_COLS] = True
    box_flat = box.reshape(-1)
    chance = box_flat.mean()
    print("\n" + "=" * 78)
    print(f"TRIGGER LOCALIZATION top-1 hit-rate (chance={chance*100:.1f}%), per-DoF maps pooled")
    print("=" * 78)
    print(f"{'map type':<28} | {'POISON hit%':>12} | {'clean hit%':>11}")
    print("-" * 60)
    for name, key in [("weight  L27", "maps_w_L27"), ("norm    L27", "maps_n_L27"),
                      ("weight  L31", "maps_w_L31"), ("norm    L31", "maps_n_L31")]:
        maps = save[key]                                     # (N, 7, 256)
        hit = box_flat[maps.reshape(len(roles), -1, maps.shape[-1]).mean(axis=1).argmax(axis=1)]
        print(f"{name:<28} | {hit[is_p].mean()*100:>11.1f}% | {hit[ct].mean()*100:>10.1f}%")
    att = save["attrib"]
    hit = box_flat[np.clip(att, 0, None).mean(axis=1).argmax(axis=1)]
    print(f"{'logit-attribution L31 (+)':<28} | {hit[is_p].mean()*100:>11.1f}% | {hit[ct].mean()*100:>10.1f}%")

    # ---- attribution concentration as a detector ----
    print("\nACTION-LOGIT ATTRIBUTION detector (share of |attribution| on argmax patch):")
    print(f"{'statistic':<26} | {'poison':>8} | {'ketchup':>8} | {'milk':>7} | {'spec. gap':>10}")
    print("-" * 72)
    a = np.clip(att, 0, None).mean(axis=1)                   # positive contributions, DoF-pooled
    share = a.max(axis=1) / np.clip(a.sum(axis=1), 1e-12, None)
    report_row("attrib max-patch share", share)
    tot = np.abs(att).sum(axis=(1, 2))
    report_row("attrib total magnitude", tot)


if __name__ == "__main__":
    main()
