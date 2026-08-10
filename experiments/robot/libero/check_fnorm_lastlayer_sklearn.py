"""
experiments/robot/libero/check_fnorm_lastlayer_sklearn.py

AUROC for the last-layer-only f_norm detector, via sklearn.roc_auc_score.

Reads the .npz produced by the last-layer-only mode of
run_attention_assimilation_detector.py (assimilation_stats_lastlayer_fnorm.npz),
where f_norm is already a flat 1D array -- one value per sample, since only the
last layer was ever computed (no per-layer dimension to slice into anymore).

Usage: python check_fnorm_lastlayer_sklearn.py
"""
import os

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
NPZ = f"{REPO}/attn_maps/assimilation_stats_lastlayer_fnorm.npz"


def main():
    d = np.load(NPZ, allow_pickle=True)
    cond = d["cond"].astype(str)
    labels = (cond == "poison").astype(int)   # 1 = poison, 0 = clean

    task_id = d["task_id"]
    seed = d["seed"]
    print("Tasks run:", sorted(set(task_id.tolist())), f"({len(set(task_id.tolist()))} tasks)")
    print("Seeds run:", sorted(set(seed.tolist())), f"({len(set(seed.tolist()))} seeds)")
    print(f"samples: {len(labels)}  (poison={labels.sum()}, clean={(1 - labels).sum()})")

    last_layer_vals = d["f_norm"]  # already 1D, shape (n_samples,) -- last-layer-only mode
    print(f"f_norm value range: [{last_layer_vals.min():.4f}, {last_layer_vals.max():.4f}]")
    print(f"  clean mean:  {last_layer_vals[labels == 0].mean():.4f}")
    print(f"  poison mean: {last_layer_vals[labels == 1].mean():.4f}")

    # sklearn expects "higher score = more positive (poison)". f_norm is expected to go
    # DOWN under poison (assimilation), so the raw AUROC should come out BELOW 0.5 --
    # report both the raw and the sign-flipped version.
    raw_auroc = roc_auc_score(labels, last_layer_vals)
    if raw_auroc >= 0.5:
        directed_auroc, direction = raw_auroc, "up"
    else:
        directed_auroc, direction = 1.0 - raw_auroc, "down"

    print(f"\nsklearn roc_auc_score (raw, higher f_norm = more 'poison-like'): {raw_auroc:.4f}")
    print(f"direction-corrected AUROC: {directed_auroc:.4f}  (direction={direction})")


if __name__ == "__main__":
    main()
