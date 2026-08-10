"""
experiments/robot/libero/analyze_trigger_localization.py

Can the per-DoF action attention LOCALIZE the trigger, not just flag it?

Detection tells an operator "something is wrong". Localization tells them WHERE,
which is what makes a defense actionable on a robot (mask that region, re-plan,
or refuse to act). We test whether the argmax patch of each DoF's last-layer
attention over the 16x16 grid lands inside the trigger object's footprint.

IMPORTANT -- the trigger box is used ONLY as evaluation ground truth here, never
as an input to any detector. run_attention_assimilation_detector.py's `trig_mass`
statistic consumed this box as a *feature*, which made it an oracle and not
deployable; this script inverts that relationship: the detector proposes a
location from attention alone, and the box merely scores whether the proposal
was right. Hit-rate on CLEAN scenes is reported alongside as the null -- clean
scenes have no trigger, so any "hits" there are the chance rate of that box.

Reads attn_maps/attention_action_coupling.npz (needs the img_maps field).

Usage: python analyze_trigger_localization.py
"""
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
NPZ = f"{REPO}/attn_maps/attention_action_coupling.npz"

GRID = 16
# Approximate patch footprint of the GoBA trigger object, same box used as the
# oracle feature in run_attention_assimilation_detector.py -- here it is GROUND
# TRUTH ONLY (scores predictions; never fed to the predictor).
TRIG_ROWS, TRIG_COLS = slice(11, 15), slice(0, 4)
DOF_NAMES = ["dx", "dy", "dz", "roll", "pitch", "yaw", "gripper"]


def main():
    d = np.load(NPZ, allow_pickle=True)
    cond = d["cond"].astype(str)
    maps = d["img_maps"]            # (N, 7, 256)
    n, n_dof, n_patch = maps.shape
    print(f"samples={n}  dofs={n_dof}  patches={n_patch}")

    # Boolean mask of the ground-truth trigger footprint, flattened to patch indices.
    box = np.zeros((GRID, GRID), dtype=bool)
    box[TRIG_ROWS, TRIG_COLS] = True
    box_flat = box.reshape(-1)
    chance = box_flat.mean()
    print(f"trigger box covers {box_flat.sum()}/{n_patch} patches "
          f"-> {chance*100:.1f}% hit-rate expected from random guessing\n")

    is_p, is_c = cond == "poison", cond == "clean"

    print("Top-1 localization hit-rate: does the argmax attention patch fall in the trigger box?")
    print(f"{'DoF':<8} | {'POISON hit%':>12} | {'clean hit%':>11} | {'lift over chance':>17}")
    print("-" * 60)
    for i, name in enumerate(DOF_NAMES):
        am = maps[:, i, :].argmax(axis=1)
        hit = box_flat[am]
        hp, hc = hit[is_p].mean(), hit[is_c].mean()
        print(f"{name:<8} | {hp*100:>11.1f}% | {hc*100:>10.1f}% | {hp/chance:>16.1f}x")

    # Pooling all 7 DoFs into one consensus map per sample (the natural "one
    # localization per observation" a deployed system would actually use).
    pooled = maps.mean(axis=1)
    am = pooled.argmax(axis=1)
    hit = box_flat[am]
    hp, hc = hit[is_p].mean(), hit[is_c].mean()
    print(f"\n{'POOLED':<8} | {hp*100:>11.1f}% | {hc*100:>10.1f}% | {hp/chance:>16.1f}x")

    # Mass-based view: what share of the attention distribution sits on the trigger?
    print("\nAttention mass inside the trigger box (mean over samples):")
    print(f"{'DoF':<8} | {'poison mass':>12} | {'clean mass':>11} | {'ratio':>7}")
    print("-" * 48)
    for i, name in enumerate(DOF_NAMES):
        m = maps[:, i, :]
        frac = m[:, box_flat].sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        fp, fc = frac[is_p].mean(), frac[is_c].mean()
        print(f"{name:<8} | {fp:>12.4f} | {fc:>11.4f} | {fp/max(fc,1e-9):>6.2f}x")


if __name__ == "__main__":
    main()
