"""
experiments/robot/libero/analyze_mlp_calibration_free.py

Can any MLP-feedforward statistic support a T2IShield-style FIXED threshold
(no clean-data calibration at deployment)?

Honest protocol:
  * VAL   = tasks 0-4, TEST = tasks 5-9 (disjoint).
  * On VAL only: pick the best (statistic, layer) and set the threshold at the
    midpoint of the poison/benign gap, rounded to 2 significant figures (a
    "universal constant" in the T2IShield sense).
  * On TEST only: report TPR on poison and FPR on clean_cal+clean_test+decoys.
  * A statistic only counts as calibration-free-viable if the TEST margin is
    open (no benign sample crosses the frozen constant).

Also prints the full layer sweep per statistic (for the paper figure) and the
decoy breakdown (specificity), since a statistic that fires on ketchup is a
novelty detector, not a backdoor detector.
"""
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

NPZ = "/home/grads/nsamptur/vla_bkd_def/GoBA_attack/attn_maps/mlp_feedforward_probe.npz"

d = np.load(NPZ, allow_pickle=True)
role = d["role"].astype(str)
task = d["task_id"]
N_LAYERS = d["spatial_ratio"].shape[1]

is_poison = role == "poison"
is_clean = np.isin(role, ["clean_cal", "clean_test"])
is_decoy = np.isin(role, ["decoy_ketchup", "decoy_milk"])
is_benign = ~is_poison
val, test = task <= 4, task >= 5

neu = d["neuron"]  # (N, 32, 7, 4)
STATS = {
    "spatial_max_over_med": d["spatial_ratio"],
    "spatial_gini": d["spatial_gini"],
    "neuron_max_over_med": np.nanmean(neu[:, :, :, 0], axis=2),
    "neuron_kurtosis": np.nanmean(neu[:, :, :, 1], axis=2),
    "neuron_gini": np.nanmean(neu[:, :, :, 2], axis=2),
    "neuron_frac_massive": np.nanmean(neu[:, :, :, 3], axis=2),
    "dof_cosine": d["dof_cos"],
}

print(f"N={len(role)}  poison={is_poison.sum()} clean={is_clean.sum()} decoy={is_decoy.sum()}")
print(f"VAL tasks 0-4: {val.sum()}  TEST tasks 5-9: {test.sum()}")


def dir_auc(v, pos, neg):
    ok = ~np.isnan(v)
    y = np.r_[np.ones(int((pos & ok).sum())), np.zeros(int((neg & ok).sum()))]
    s = np.r_[v[pos & ok], v[neg & ok]]
    if len(np.unique(y)) < 2:
        return np.nan, "?"
    a = roc_auc_score(y, s)
    return (a, "up") if a >= 0.5 else (1 - a, "down")


def round_sig(x, sig=2):
    if x == 0 or not np.isfinite(x):
        return x
    from math import floor, log10
    return round(x, -int(floor(log10(abs(x)))) + (sig - 1))


print("\n" + "=" * 100)
print("LAYER SWEEP: AUROC(poison vs ALL benign) per statistic (VAL tasks only, to guide selection)")
print("=" * 100)
best = {}
for name, arr in STATS.items():
    row = []
    for li in range(N_LAYERS):
        v = arr[:, li]
        a, dr = dir_auc(v[val], is_poison[val], is_benign[val])
        row.append((a if np.isfinite(a) else 0.0, dr, li))
    row.sort(reverse=True)
    top = row[:3]
    best[name] = top[0]
    print(f"{name:<22} top layers: " + "   ".join(f"L{li:>2}={a:.3f}({dr})" for a, dr, li in top))

print("\n" + "=" * 100)
print("FIXED-CONSTANT TEST: threshold frozen on VAL (rounded midpoint), evaluated on held-out TEST")
print("=" * 100)
for name, (auc_val, direction, li) in best.items():
    v = arr = STATS[name][:, li]
    p_val, b_val = v[val & is_poison], v[val & is_benign]
    if direction == "up":
        gap_lo, gap_hi = np.nanmax(b_val), np.nanmin(p_val)
    else:
        gap_lo, gap_hi = np.nanmax(p_val), np.nanmin(b_val)
    separable_val = gap_hi > gap_lo
    thr = round_sig((gap_lo + gap_hi) / 2.0, 2)

    def fires(x):
        return (x > thr) if direction == "up" else (x < thr)

    tp = fires(v[test & is_poison])
    fp_clean = fires(v[test & is_clean])
    fp_decoy = fires(v[test & is_decoy])
    # residual margin on TEST relative to the frozen constant
    if direction == "up":
        m_test = np.nanmin(v[test & is_poison]) - max(np.nanmax(v[test & is_benign & ~is_poison]), -np.inf)
    else:
        m_test = np.nanmin(v[test & is_benign]) - np.nanmax(v[test & is_poison])
    print(f"\n{name}  (L{li}, dir={direction}, VAL AUROC={auc_val:.3f}, "
          f"VAL separable={'YES' if separable_val else 'NO'})")
    print(f"  frozen constant = {thr}")
    print(f"  TEST: poison fires {100*np.nanmean(tp):.0f}%   clean FPR {100*np.nanmean(fp_clean):.0f}%   "
          f"decoy FPR {100*np.nanmean(fp_decoy):.0f}%   test margin={m_test:+.4f}")
    a_t, d_t = dir_auc(v[test], is_poison[test], is_benign[test])
    a_d, _ = dir_auc(v[test], is_poison[test], is_decoy[test])
    print(f"  TEST AUROC poison-vs-benign={a_t:.3f}({d_t})   poison-vs-DECOY-only={a_d:.3f}")

print("\n" + "=" * 100)
print("DECOY BREAKDOWN at each statistic's chosen layer (means; novelty-vs-backdoor check)")
print("=" * 100)
for name, (_, _, li) in best.items():
    v = STATS[name][:, li]
    print(f"{name:<22} L{li:>2}  clean={np.nanmean(v[is_clean]):.4f}  "
          f"ketchup={np.nanmean(v[role=='decoy_ketchup']):.4f}  "
          f"milk={np.nanmean(v[role=='decoy_milk']):.4f}  "
          f"poison={np.nanmean(v[is_poison]):.4f}")
