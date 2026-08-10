#!/usr/bin/env python3
"""Score candidate action->image map statistics as backdoor detectors.

Each statistic is evaluated on the 2x2 design (model in {clean, backdoored} x
input in {clean, poison}).  A useful detector must separate clean from poison on
BOTH backdoored models while staying at chance on BOTH matched clean models --
otherwise it is measuring the trigger's visual footprint, not the backdoor.

Usage:
    python score_map_stats_qktv_temporal.py                # ranked table
    python score_map_stats_qktv_temporal.py --per-t        # + per-timestep AUROC
    python score_map_stats_qktv_temporal.py --ablate       # + top-K masking ablation
    python score_map_stats_qktv_temporal.py --fingerprint  # + weights-only (clean input) test
    python score_map_stats_qktv_temporal.py --all
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

OUT = Path("/home/grads/nsamptur/vla_bkd_def/GoBA_attack/attn_maps/action_imgtext_qktv_temporal")
GRID = 16

# (backdoored model, matched clean control) -- the control uses the same trigger protocol.
PAIRS = [("goba", "clean_goba_protocol"), ("badvla", "clean_badvla_protocol")]
MODELS = [m for pair in PAIRS for m in pair]
SHORT = {
    "goba": "GoBA",
    "clean_goba_protocol": "cOpenVLA",
    "badvla": "BadVLA",
    "clean_badvla_protocol": "cOFT",
}


def load(name):
    d = np.load(OUT / f"{name}_temporal.npz", allow_pickle=True)
    return {
        "cond": np.array([str(c) for c in d["cond"]]),
        "task_id": d["task_id"],
        "maps_w": d["maps_weight"],  # (E, T, 256), rows sum to 1
        "maps_q": d["maps_qktv"],
        "ftt_w": d["ftt_weight"],  # (E, T)
        "ftt_q": d["ftt_qktv"],
    }


# --------------------------------------------------------------------------- #
# map statistics.  Each takes the loaded dict and returns (E,) or (E, T), with
# the convention that HIGHER = more poison-like, so AUROC is always P(poison >
# clean) and 1.0 means perfect detection.
# --------------------------------------------------------------------------- #
def _moran(maps, mask_top=0):
    """Unnormalised 4-neighbour spatial autocorrelation of the patch grid.

    This is sum_adj(c_i c_j) / sum(c_i^2) without the textbook N/W factor;
    multiply by 0.533 for Moran's I proper.  The rescaling is monotone, so it
    does not affect AUROC.  mask_top drops the K highest patches first, which
    separates "one contiguous hot blob" from "the whole map is smooth".
    """
    if mask_top:
        rank = (-maps).argsort(-1).argsort(-1)
        maps = np.where(rank < mask_top, np.nan, maps)
    g = maps.reshape(*maps.shape[:-1], GRID, GRID)
    c = np.nan_to_num(g - np.nanmean(g, (-1, -2), keepdims=True))
    num = (c[..., :, :-1] * c[..., :, 1:]).sum((-1, -2)) + (c[..., :-1, :] * c[..., 1:, :]).sum((-1, -2))
    return num / np.clip((c**2).sum((-1, -2)), 1e-20, None)


def _blob(maps, top_k=16):
    """Size of the largest 4-connected component among the top-K patches."""
    rank = (-maps).argsort(-1).argsort(-1)
    hot = (rank < top_k).reshape(*maps.shape[:-1], GRID, GRID)
    out = np.zeros(maps.shape[:-1])
    for idx in np.ndindex(*maps.shape[:-1]):
        lab, n = ndimage.label(hot[idx])
        out[idx] = max((lab == i).sum() for i in range(1, n + 1)) if n else 0
    return out


def _entropy(maps):
    p = np.clip(maps, 1e-12, None)
    p = p / p.sum(-1, keepdims=True)
    return -(p * np.log(p)).sum(-1) / np.log(256)


def _radius(maps):
    g = maps.reshape(*maps.shape[:-1], GRID, GRID)
    yy, xx = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    cy, cx = (g * yy).sum((-1, -2)), (g * xx).sum((-1, -2))
    return np.hypot(cy - 7.5, cx - 7.5)


def _js(a, b):
    a, b = a / a.sum(-1, keepdims=True), b / b.sum(-1, keepdims=True)
    m = 0.5 * (a + b)
    kl = lambda x, y: (x * np.log(np.clip(x, 1e-12, None) / np.clip(y, 1e-12, None))).sum(-1)
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


STATS = {
    # baselines from the original analysis
    "ftt_qktv": lambda d: -d["ftt_q"],
    "ftt_weight": lambda d: -d["ftt_w"],
    # spatial autocorrelation -- the strongest unified statistic found
    "moran_w": lambda d: _moran(d["maps_w"]),
    "moran_q": lambda d: _moran(d["maps_q"]),
    # mechanism-matched decomposition of moran
    "blob16_w": lambda d: _blob(d["maps_w"], 16),  # GoBA-shaped: hot-blob extent
    "moran_w_mask16": lambda d: _moran(d["maps_w"], mask_top=16),  # BadVLA-shaped: background smoothness
    # informative but weaker / confounded -- kept so the caveats stay visible
    "entropy_q": lambda d: _entropy(d["maps_q"]),
    "radius_q": lambda d: _radius(d["maps_q"]),
    "wq_js": lambda d: _js(d["maps_w"], d["maps_q"]),
}

NOTES = {
    "entropy_q": "sign flips between attacks (BadVLA up, GoBA down) -- not unified",
    "radius_q": "encodes where this particular trigger sits; will not transfer",
    "blob16_w": "GoBA-specific; near chance on BadVLA",
    "moran_w_mask16": "BadVLA-specific; reverses on GoBA",
}


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def auroc(pos, neg):
    return float(np.mean([[(a > b) + 0.5 * (a == b) for b in neg] for a in pos]))


def bootstrap_ci(pos, neg, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [auroc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def episode_scores(fn, data):
    v = np.asarray(fn(data), dtype=float)
    return v.mean(1) if v.ndim == 2 else v


def evaluate(fn, data, n_boot=4000):
    v = episode_scores(fn, data)
    cond = data["cond"]
    pos, neg = v[cond == "poison"], v[cond == "clean"]
    lo, hi = bootstrap_ci(pos, neg, n_boot)
    return {
        "auroc": auroc(pos, neg),
        "ci": (lo, hi),
        "open": bool(min(pos) > max(neg) or min(neg) > max(pos)),
        "clean": neg.mean(),
        "poison": pos.mean(),
    }


def rank_table(loaded, n_boot):
    print("AUROC = P(poison > clean); * = open margin (no overlap).")
    print("SCORE = min separation on the two backdoors - max separation on the two clean controls.\n")
    head = f"{'statistic':18s}" + "".join(f"{SHORT[m]:>11s}" for m in MODELS) + f"{'both':>7s}{'ctrl':>7s}{'SCORE':>8s}"
    print(head)
    print("-" * len(head))

    rows = []
    for name, fn in STATS.items():
        res = {m: evaluate(fn, loaded[m], n_boot) for m in MODELS}
        sep = lambda a: abs(a - 0.5) * 2
        bd = [res[b]["auroc"] for b, _ in PAIRS]
        ctl = [res[c]["auroc"] for _, c in PAIRS]
        same = len({np.sign(a - 0.5) for a in bd}) == 1
        both = min(sep(a) for a in bd) if same else 0.0
        ctrl = max(sep(a) for a in ctl)
        rows.append((both - ctrl, name, res, both, ctrl))

    for score, name, res, both, ctrl in sorted(rows, key=lambda r: -r[0]):
        line = f"{name:18s}"
        for m in MODELS:
            line += f"{res[m]['auroc']:>10.2f}" + ("*" if res[m]["open"] else " ")
        print(line + f"{both:>7.2f}{ctrl:>7.2f}{score:>8.2f}")

    print("\nBootstrap 95% CIs on the backdoored models:")
    for score, name, res, _, _ in sorted(rows, key=lambda r: -r[0]):
        parts = [f"{SHORT[b]} {res[b]['auroc']:.2f} [{res[b]['ci'][0]:.2f},{res[b]['ci'][1]:.2f}]" for b, _ in PAIRS]
        note = f"   <- {NOTES[name]}" if name in NOTES else ""
        print(f"  {name:18s} " + "   ".join(parts) + note)


def per_timestep(loaded, names):
    print("\nPer-timestep AUROC (single frame, no temporal pooling):")
    for name in names:
        print(f"  [{name}]")
        for m in MODELS:
            d = loaded[m]
            v = np.asarray(STATS[name](d), dtype=float)
            if v.ndim == 1:
                print(f"    {SHORT[m]:9s} (episode-level statistic, no per-t decomposition)")
                continue
            cond = d["cond"]
            per_t = [auroc(v[cond == "poison"][:, t], v[cond == "clean"][:, t]) for t in range(v.shape[1])]
            print(f"    {SHORT[m]:9s} " + " ".join(f"{a:.2f}" for a in per_t))


def ablate(loaded, ks=(0, 1, 4, 9, 16, 32, 64)):
    print("\nTop-K masking ablation on moran_w -- does the signal survive removing the hot patches?")
    print("  (collapse => the statistic was measuring one contiguous blob; survival => global structure)")
    print(f"    {'mask top':>10s}" + "".join(f"{SHORT[m]:>11s}" for m in MODELS))
    for k in ks:
        line = f"    {k:>10d}"
        for m in MODELS:
            d = loaded[m]
            v = _moran(d["maps_w"], mask_top=k).mean(1)
            line += f"{auroc(v[d['cond'] == 'poison'], v[d['cond'] == 'clean']):>11.2f}"
        print(line)


def fingerprint(loaded, n_boot):
    print("\nWeights-only fingerprint: backdoored vs matched clean checkpoint, CLEAN INPUT ONLY.")
    print("  (a separation here means the backdoor is detectable without ever presenting the trigger)")
    for bd, cl in PAIRS:
        for name in ("moran_w", "ftt_qktv"):
            fn = STATS[name]
            a = episode_scores(fn, loaded[bd])[loaded[bd]["cond"] == "clean"]
            b = episode_scores(fn, loaded[cl])[loaded[cl]["cond"] == "clean"]
            lo, hi = bootstrap_ci(a, b, n_boot)
            flag = "OPEN MARGIN" if (min(a) > max(b) or min(b) > max(a)) else "overlap"
            print(
                f"    {SHORT[bd]:8s} vs {SHORT[cl]:9s} {name:15s} "
                f"bd={a.mean():+.3f} clean={b.mean():+.3f}  "
                f"AUROC={auroc(a, b):.2f} CI[{lo:.2f},{hi:.2f}]  {flag}"
            )
    print("\n  Caveat: two checkpoints at n=10 tasks each. Establishing that ordinary finetuning")
    print("  does not also move this statistic needs several independent clean checkpoints.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-t", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--fingerprint", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=4000)
    args = ap.parse_args()

    loaded = {m: load(m) for m in MODELS}
    n_ep = len(loaded[MODELS[0]]["cond"])
    print(f"Loaded {len(MODELS)} models x {n_ep} episodes ({n_ep // 2} clean / {n_ep // 2} poison), "
          f"T={loaded[MODELS[0]]['maps_w'].shape[1]}\n")

    rank_table(loaded, args.bootstrap)
    if args.per_t or args.all:
        per_timestep(loaded, ["moran_w", "ftt_qktv"])
    if args.ablate or args.all:
        ablate(loaded)
    if args.fingerprint or args.all:
        fingerprint(loaded, args.bootstrap)

    print("\nLimitation: only DoF-averaged 256-vectors are stored, so no DoF-resolved competitor")
    print("to FTT (per-DoF moran, DoF rank agreement, cross-DoF Gram spectrum) can be tested here.")
    print("Saving the full 7x256 tensor and the pre-renormalisation image/text mass split would")
    print("allow a proper head-to-head.")


if __name__ == "__main__":
    main()
