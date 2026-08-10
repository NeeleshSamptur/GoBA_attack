#!/usr/bin/env python3
"""Confirmatory validation of ONE pre-declared statistic across seeds, splits and attacks.

Primary statistic:   FTT -- Frobenius/L2 dispersion of the 7 DoF rows of the
                     action->image attention matrix about their mean row.
Secondary statistic: moran -- 4-neighbour spatial autocorrelation of the
                     DoF-averaged map (reported for the files that store maps).

No statistic selection happens in this script.  Both statistics were fixed
before any of these datasets were opened; everything below is out-of-sample
with respect to the exploratory sweep run on
`attn_maps/action_imgtext_qktv_temporal/*_temporal.npz` (seed 7 only).

Usage: python validate_ftt_across_seeds.py
"""
from pathlib import Path

import numpy as np

ATTN = Path("/home/grads/nsamptur/vla_bkd_def/GoBA_attack/attn_maps")
G = 16
rng = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# statistics (fixed -- do not add to this section)
# --------------------------------------------------------------------------- #
def ftt(maps):
    """maps (..., 7, 256) -> (...) mean L2 distance of DoF rows from their mean row."""
    p = maps / np.clip(maps.sum(-1, keepdims=True), 1e-20, None)
    return np.linalg.norm(p - p.mean(-2, keepdims=True), axis=-1).mean(-1)


def moran(maps):
    """maps (..., 7, 256) -> (...) spatial autocorrelation of the DoF-averaged map."""
    m = maps.mean(-2)
    m = m / np.clip(m.sum(-1, keepdims=True), 1e-20, None)
    g = m.reshape(*m.shape[:-1], G, G)
    c = g - g.mean((-1, -2), keepdims=True)
    num = (c[..., :, :-1] * c[..., :, 1:]).sum((-1, -2)) + (c[..., :-1, :] * c[..., 1:, :]).sum((-1, -2))
    return num / np.clip((c ** 2).sum((-1, -2)), 1e-20, None)


# --------------------------------------------------------------------------- #
def auroc(pos, neg):
    """Mann-Whitney U form -- O(n log n) instead of the O(n^2) double loop."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    r = np.empty_like(allv)
    order = allv.argsort()
    sv = allv[order]
    ranks = np.arange(1, allv.size + 1, dtype=float)
    # average ranks within ties
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[i:j + 1] = ranks[i:j + 1].mean()
        i = j + 1
    r[order] = ranks
    n1 = pos.size
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * neg.size))


def boot_ci(pos, neg, n=2000):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size < 2 or neg.size < 2:
        return (np.nan, np.nan)
    P = rng.choice(pos, (n, pos.size))
    N = rng.choice(neg, (n, neg.size))
    # vectorised AUROC: fraction of (p, q) pairs with p > q, per bootstrap draw
    v = ((P[:, :, None] > N[:, None, :]).mean((1, 2))
         + 0.5 * (P[:, :, None] == N[:, None, :]).mean((1, 2)))
    return tuple(np.percentile(v, [2.5, 97.5]))


def strarr(a):
    return np.array([str(x) for x in a])


def report(name, pos, neg, direction="poison<clean"):
    """direction says which way the statistic is expected to move under trigger."""
    a, b = (neg, pos) if direction == "poison<clean" else (pos, neg)
    A = auroc(a, b)
    lo, hi = boot_ci(a, b)
    ov = "open" if (len(a) and len(b) and min(a) > max(b)) else ""
    print(f"    {name:34s} n={len(pos):3d}/{len(neg):3d}  clean={np.mean(neg):+.4f} "
          f"poison={np.mean(pos):+.4f}  AUROC={A:.2f} CI[{lo:.2f},{hi:.2f}] {ov}")
    return A


# --------------------------------------------------------------------------- #
# 1. f_norm across 20 seeds, backdoored vs clean model
# --------------------------------------------------------------------------- #
def seed_sweep():
    print("=" * 78)
    print("1. PRE-COMPUTED f_norm (last layer): 20 seeds x 10 tasks, BD vs clean model")
    print("=" * 78)
    bd = np.load(ATTN / "assimilation_stats_lastlayer_fnorm.npz", allow_pickle=True)
    cm = np.load(ATTN / "assimilation_stats_alllayers_fnorm_CLEANMODEL.npz", allow_pickle=True)

    for tag, d in [("BACKDOORED", bd), ("CLEAN MODEL (control)", cm)]:
        f = d["f_norm"]
        f = f[:, -1] if f.ndim == 2 else f
        cond, seed = strarr(d["cond"]), d["seed"]
        print(f"\n  [{tag}]  pooled over all seeds:")
        report("f_norm pooled", f[cond == "poison"], f[cond == "clean"])

        per = []
        for s in sorted(set(seed.tolist())):
            m = seed == s
            p, c = f[m & (cond == "poison")], f[m & (cond == "clean")]
            if len(p) and len(c):
                per.append(auroc(c, p))
        if per:
            per = np.array(per)
            print(f"    per-seed AUROC over {len(per)} seeds: mean={per.mean():.2f} "
                  f"median={np.median(per):.2f} min={per.min():.2f} max={per.max():.2f}")
            print(f"    seeds with AUROC>=0.9: {(per >= 0.9).sum()}/{len(per)}   "
                  f">=0.8: {(per >= 0.8).sum()}/{len(per)}   <=0.6: {(per <= 0.6).sum()}/{len(per)}")
        else:
            # conditions live in paired seeds rather than within a seed
            seeds = sorted(set(seed.tolist()))
            cl = [s for s in seeds if (cond[seed == s] == "clean").all()]
            po = [s for s in seeds if (cond[seed == s] == "poison").all()]
            print(f"    conditions are seed-paired: {len(cl)} clean seeds, {len(po)} poison seeds")
            per = []
            for sc, sp in zip(sorted(cl), sorted(po)):
                per.append(auroc(f[seed == sc], f[seed == sp]))
            per = np.array(per)
            print(f"    per seed-pair AUROC over {len(per)} pairs: mean={per.mean():.2f} "
                  f"median={np.median(per):.2f} min={per.min():.2f} max={per.max():.2f}")
            print(f"    pairs with AUROC>=0.9: {(per >= 0.9).sum()}/{len(per)}   "
                  f">=0.8: {(per >= 0.8).sum()}/{len(per)}   <=0.6: {(per <= 0.6).sum()}/{len(per)}")


# --------------------------------------------------------------------------- #
# 2. held-out task split
# --------------------------------------------------------------------------- #
def task_disjoint():
    print("\n" + "=" * 78)
    print("2. TASK-DISJOINT split (held-out tasks, seeds 5..424242)")
    print("=" * 78)
    d = np.load(ATTN / "assimilation_stats_taskdisjoint_fnorm.npz", allow_pickle=True)
    f = d["f_norm"]
    f = f[:, -1] if f.ndim == 2 else f
    cond, tid = strarr(d["cond"]), d["task_id"]
    print()
    report("f_norm all held-out tasks", f[cond == "poison"], f[cond == "clean"])
    for half, name in [(range(0, 5), "tasks 0-4"), (range(5, 10), "tasks 5-9")]:
        m = np.isin(tid, list(half))
        report(f"f_norm {name}", f[m & (cond == "poison")], f[m & (cond == "clean")])


# --------------------------------------------------------------------------- #
# 3. trajectory DoF maps: recompute FTT + moran from raw 7x256, with decoys
# --------------------------------------------------------------------------- #
def trajectory_files():
    print("\n" + "=" * 78)
    print("3. RAW 7x256 DoF MAPS -- FTT and moran recomputed from scratch")
    print("=" * 78)
    files = [
        ("GoBA BD", "trajectory_dof_attention.npz"),
        ("Clean model", "cleanmodel_trajectory_dof_attention.npz"),
        ("BadVLA BD", "badvla_trajectory_dof_attention.npz"),
        ("AttackVLA BD (3rd attack)", "attackvla_trajectory_dof_attention.npz"),
    ]
    for label, fn in files:
        p = ATTN / fn
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        maps = d["maps"]                       # (E, T, 7, 256)
        role, seed = strarr(d["role"]), d["seed"]
        F = ftt(maps).mean(1)
        M = moran(maps).mean(1)
        roles = sorted(set(role.tolist()))
        base = "clean"
        print(f"\n  [{label}]  {fn}  roles={roles}  seeds={sorted(set(seed.tolist()))}")
        for r in roles:
            if r == base:
                continue
            print(f"    -- {base} vs {r} --")
            report(f"FTT   ({r})", F[role == r], F[role == base])
            report(f"moran ({r})", M[role == r], M[role == base], direction="poison>clean")
        # per-seed stability of the primary statistic on the main poison role
        main = "poison" if "poison" in roles else ("full_trigger" if "full_trigger" in roles else None)
        if main:
            per = []
            for s in sorted(set(seed.tolist())):
                m = seed == s
                a, b = F[m & (role == main)], F[m & (role == base)]
                if len(a) and len(b):
                    per.append(auroc(b, a))
            if per:
                print(f"    FTT per-seed AUROC ({base} vs {main}): "
                      + " ".join(f"{x:.2f}" for x in per))


# --------------------------------------------------------------------------- #
def main():
    seed_sweep()
    task_disjoint()
    trajectory_files()
    print("\n" + "=" * 78)
    print("Read: AUROC is oriented so that >0.5 means the statistic moves in the")
    print("expected direction under trigger (FTT down, moran up). A clean model must")
    print("sit near 0.50; a real detector must hold up on held-out tasks and seeds.")
    print("=" * 78)


if __name__ == "__main__":
    main()
