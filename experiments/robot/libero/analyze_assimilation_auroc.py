"""
AUROC analysis for the training-free attention-assimilation detectors.

Reads the .npz produced by run_attention_assimilation_detector.py and reports,
per statistic and per layer, how well clean vs. trigger scenes separate.

AUROC here is computed via the Mann-Whitney U identity (no sklearn dependency),
and is direction-corrected: a statistic that goes DOWN under trigger gets scored
on its negation, so 0.5 always means "no separation" and 1.0 means "perfect",
regardless of sign. The reported `direction` column says which way it moved.

Also reports a task-held-out estimate: because layer choice is itself a fitted
decision, picking the best layer on all the data overstates performance. The
leave-one-task-out column picks the best layer using 9 tasks and evaluates on
the held-out one, which is the honest number to quote.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
NPZ = f"{REPO}/attn_maps/assimilation_stats.npz"

STAT_NAMES = [
    "f_norm",
    "mean_pairwise_cos",
    "consensus_entropy",
    "mean_token_entropy",
    "mean_token_max",
    "trig_mass",
]
ORACLE = {"trig_mass"}


def auroc(scores, labels):
    """labels: 1 = positive (poison). Returns P(score_pos > score_neg) + 0.5*ties."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    arr = np.concatenate([pos, neg])[order]
    i = 0
    r = np.empty(len(arr))
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[j + 1] == arr[i]:
            j += 1
        r[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    ranks[order] = r
    rank_pos = ranks[: len(pos)].sum()
    return (rank_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def directed_auroc(values, labels):
    """Returns (auroc_after_direction_correction, direction_string)."""
    a = auroc(values, labels)
    if a >= 0.5:
        return a, "up"
    return 1.0 - a, "down"


def main():
    if not os.path.exists(NPZ):
        sys.exit(f"missing {NPZ} -- run run_attention_assimilation_detector.py first")
    d = np.load(NPZ, allow_pickle=True)
    cond = d["cond"].astype(str)
    conds_present = sorted(set(cond.tolist()))
    n_layers = d[STAT_NAMES[0]].shape[1]
    print(f"conditions present: {conds_present}")
    for c in conds_present:
        print(f"  {c}: n={int((cond==c).sum())}")

    labels = (cond == "poison").astype(int)
    task_id = d["task_id"]

    print(f"samples: {len(labels)}  (poison={labels.sum()}, clean={(1-labels).sum()})  layers: {n_layers}")

    # sanity: did seeds actually vary the scenes?
    chk = d["frame_checksum"]
    for c in ("clean", "poison"):
        v = chk[cond == c]
        print(f"  frame_checksum[{c}]: {len(np.unique(np.round(v,6)))} unique of {len(v)}")

    print("\n=== Best layer per statistic (selected on ALL data -- optimistic) ===")
    print(f"{'statistic':22s} {'best layer':>11s} {'AUROC':>8s} {'dir':>6s}   {'clean mean':>11s} {'poison mean':>12s}")
    best_layer_all = {}
    for stat in STAT_NAMES:
        vals = d[stat]
        per_layer = [directed_auroc(vals[:, l], labels) for l in range(n_layers)]
        aurocs = np.array([x[0] for x in per_layer])
        bl = int(np.nanargmax(aurocs))
        best_layer_all[stat] = bl
        a, direction = per_layer[bl]
        cm = vals[labels == 0, bl].mean()
        pm = vals[labels == 1, bl].mean()
        tag = "  [ORACLE]" if stat in ORACLE else ""
        print(f"{stat:22s} {bl:11d} {a:8.3f} {direction:>6s}   {cm:11.4f} {pm:12.4f}{tag}")

    print("\n=== Layer-0 and last-layer AUROC (fixed layers, no selection) ===")
    print(f"{'statistic':22s} {'L0 AUROC':>10s} {'dir':>6s} {'Llast AUROC':>13s} {'dir':>6s}")
    for stat in STAT_NAMES:
        vals = d[stat]
        a0, d0 = directed_auroc(vals[:, 0], labels)
        al, dl = directed_auroc(vals[:, -1], labels)
        tag = "  [ORACLE]" if stat in ORACLE else ""
        print(f"{stat:22s} {a0:10.3f} {d0:>6s} {al:13.3f} {dl:>6s}{tag}")

    print("\n=== Leave-one-task-out (layer chosen on 9 tasks, scored on held-out task) ===")
    print(f"{'statistic':22s} {'mean AUROC':>11s} {'std':>7s}   layers picked")
    tasks = np.unique(task_id)
    for stat in STAT_NAMES:
        vals = d[stat]
        fold_aurocs = []
        picked = []
        for t in tasks:
            tr = task_id != t
            te = task_id == t
            if labels[te].sum() == 0 or (1 - labels[te]).sum() == 0:
                continue
            tr_aurocs = np.array([directed_auroc(vals[tr, l], labels[tr])[0] for l in range(n_layers)])
            bl = int(np.nanargmax(tr_aurocs))
            picked.append(bl)
            # direction also fit on train
            _, direction = directed_auroc(vals[tr, bl], labels[tr])
            s = vals[te, bl] if direction == "up" else -vals[te, bl]
            fold_aurocs.append(auroc(s, labels[te]))
        fold_aurocs = np.array(fold_aurocs, dtype=float)
        tag = "  [ORACLE]" if stat in ORACLE else ""
        uniq = sorted(set(picked))
        print(
            f"{stat:22s} {np.nanmean(fold_aurocs):11.3f} {np.nanstd(fold_aurocs):7.3f}   "
            f"{uniq if len(uniq) <= 6 else str(uniq[:6]) + '...'}{tag}"
        )


if __name__ == "__main__":
    main()
