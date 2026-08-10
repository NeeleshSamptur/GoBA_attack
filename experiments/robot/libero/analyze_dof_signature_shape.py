"""
experiments/robot/libero/analyze_dof_signature_shape.py

Is the SHAPE of the per-DoF signature an attack fingerprint?

Existing backdoor defenses reduce the model's internal state to one scalar
"anomalous?" score -- a habit inherited from image classification, where the
output really is one label. A VLA emits a structured 7-DoF action, and an
attacker's target behavior occupies specific dimensions of it. So the per-DoF
profile of the signature carries information that a scalar throws away:

  targeted attack (GoBA: "grasp the trigger object")
      -> corruption should CONCENTRATE in the DoFs that execute that behavior
  untargeted attack (BadVLA: corrupt the perception module itself)
      -> corruption has no preferred behavior, so it should be UNIFORM

We summarize "localized vs uniform" with two scale-free statistics over the
vector of 7 per-DoF directed AUROCs (each mapped to |AUROC - 0.5|, i.e. signal
strength above chance):
  * gini            0 = perfectly uniform across DoFs, higher = concentrated
  * peak_ratio      strongest DoF / mean DoF; 1.0 = flat
Both are computed on the SAME statistic for both attacks, so the comparison is
of profile shape, not of raw detectability.

Usage: python analyze_dof_signature_shape.py
"""
import os

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DOF_NAMES = ["dx", "dy", "dz", "roll", "pitch", "yaw", "gripper"]

SOURCES = [
    # (label, npz path, entropy field, note)
    ("GoBA (targeted, OpenVLA)",
     f"{REPO}/attn_maps/attention_action_coupling.npz", "entropy_image",
     "200 samples, disjoint seeds, 10 tasks, 29871-corrected"),
    ("BadVLA (untargeted, OFT)",
     f"{REPO}/attn_maps/dof_attention_entropy_badvla.npz", "entropy_image",
     "30 samples, paired seeds, 3 held-out tasks"),
]


def directed_auroc(vals, labels):
    ok = ~np.isnan(vals)
    raw = roc_auc_score(labels[ok], vals[ok])
    return raw if raw >= 0.5 else 1.0 - raw


def gini(x):
    """0 for a perfectly flat profile, ->1 as mass concentrates in one entry."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if x.sum() <= 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main():
    print("Per-DoF signature SHAPE: localized (targeted) vs uniform (untargeted)?")
    print("Strength = |directed AUROC - 0.5| (0 = chance, 0.5 = perfect).\n")
    summary = []
    for label, path, field, note in SOURCES:
        if not os.path.exists(path):
            print(f"[skip] {label}: missing {os.path.basename(path)}")
            continue
        d = np.load(path, allow_pickle=True)
        cond = d["cond"].astype(str)
        labels = (cond == "poison").astype(int)
        if labels.sum() == 0:                      # BadVLA labels its positives "trig"
            labels = (cond == "trig").astype(int)
        ent = d[field]

        strengths = np.array([abs(directed_auroc(ent[:, i], labels) - 0.5) for i in range(len(DOF_NAMES))])
        g, pr = gini(strengths), strengths.max() / max(strengths.mean(), 1e-9)
        summary.append((label, g, pr, strengths, note))

        print(f"=== {label} ===   ({note}, n={len(labels)})")
        print("   " + "  ".join(f"{n}={s:.3f}" for n, s in zip(DOF_NAMES, strengths)))
        print(f"   gini={g:.3f}   peak/mean={pr:.2f}   "
              f"argmax={DOF_NAMES[int(strengths.argmax())]}\n")

    if len(summary) == 2:
        (l1, g1, p1, _, _), (l2, g2, p2, _, _) = summary
        print("-" * 70)
        print(f"{'':34} {'gini':>8} {'peak/mean':>11}")
        print(f"{l1:<34} {g1:>8.3f} {p1:>11.2f}")
        print(f"{l2:<34} {g2:>8.3f} {p2:>11.2f}")
        verdict = ("CONSISTENT with the hypothesis (targeted more concentrated)"
                   if g1 > g2 else
                   "NOT consistent -- targeted is not more concentrated than untargeted")
        print(f"\n=> {verdict}")
        print("   Caveat: the two runs differ in architecture, sample size and split protocol;\n"
              "   this compares profile shape only and is not a matched-protocol comparison.")


if __name__ == "__main__":
    main()
