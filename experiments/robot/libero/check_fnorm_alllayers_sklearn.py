"""
experiments/robot/libero/check_fnorm_alllayers_sklearn.py

Per-layer AUROC for the f_norm detector, via sklearn.roc_auc_score directly
(no hand-rolled Mann-Whitney implementation, no layer selection/cross-validation
-- just the raw AUROC at each of the 32 layers, independently, on the full
disjoint clean/poison sample set).

Reads assimilation_stats_alllayers_fnorm_disjoint.npz, where f_norm has shape
(n_samples, n_layers).

Usage: python check_fnorm_alllayers_sklearn.py
"""
import os

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
NPZ = f"{REPO}/attn_maps/assimilation_stats_alllayers_fnorm_disjoint.npz"


def main():
    d = np.load(NPZ, allow_pickle=True)
    cond = d["cond"].astype(str)
    labels = (cond == "poison").astype(int)   # 1 = poison, 0 = clean

    f_norm = d["f_norm"]   # shape (n_samples, n_layers)
    n_samples, n_layers = f_norm.shape
    print(f"samples: {n_samples}  (poison={labels.sum()}, clean={(1 - labels).sum()})  layers: {n_layers}")

    print(f"\n{'layer':>6} {'raw AUROC':>10} {'direction':>10} {'directed AUROC':>15}   "
          f"{'clean mean':>11} {'poison mean':>12}")
    print("-" * 72)
    for l in range(n_layers):
        vals = f_norm[:, l]
        raw = roc_auc_score(labels, vals)
        if raw >= 0.5:
            directed, direction = raw, "up"
        else:
            directed, direction = 1.0 - raw, "down"
        cm = vals[labels == 0].mean()
        pm = vals[labels == 1].mean()
        print(f"{l:>6} {raw:>10.4f} {direction:>10} {directed:>15.4f}   {cm:>11.4f} {pm:>12.4f}")


if __name__ == "__main__":
    main()
