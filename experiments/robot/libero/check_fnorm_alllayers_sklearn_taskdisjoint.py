"""
experiments/robot/libero/check_fnorm_alllayers_sklearn_taskdisjoint.py

Same as check_fnorm_alllayers_sklearn.py, pointed at the task-disjoint
(tasks 0-4 clean, 5-9 poison) result instead of the seed-disjoint one.
Kept as a separate file rather than a CLI arg, matching this codebase's
existing convention (paired scripts per protocol, not flag-driven).

Usage: python check_fnorm_alllayers_sklearn_taskdisjoint.py
"""
import os

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
NPZ = f"{REPO}/attn_maps/assimilation_stats_alllayers_fnorm_taskdisjoint.npz"


def main():
    d = np.load(NPZ, allow_pickle=True)
    cond = d["cond"].astype(str)
    labels = (cond == "poison").astype(int)

    f_norm = d["f_norm"]
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
