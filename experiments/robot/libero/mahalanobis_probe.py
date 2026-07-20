"""
experiments/robot/libero/mahalanobis_probe.py

Mahalanobis-distance backdoor-DETECTION library for GoBA_attack (vanilla
OpenVLA-7b), ported from BadVLA's `trial_error/paired_probe.py`.

Only the Mahalanobis-specific pieces are ported here (NOT the L2/cosine/JS
drift-table machinery from the source file, which this repo does not need):

  * `pool_tokens`                    -- pools a raw activation tensor to 1-D.
  * `_split_indices`                 -- calibration/test scene index split.
  * `auroc`                          -- AUROC via sklearn.
  * `compute_mahalanobis_by_group`   -- the actual Mahalanobis math. Copied
                                         verbatim from BadVLA; the variance-
                                         floor logic and its scale-invariance
                                         argument are preserved exactly as
                                         written there -- do not "simplify"
                                         them, they are load-bearing.
  * `Capture` / `_make_hook` / `_to_numpy` / `_register_linears`
                                      -- hook-registration primitives. These
                                         are architecture-agnostic and are
                                         copied near-verbatim from the source.

What is NEW here (i.e. NOT a port) is `register_mahalanobis_hooks`, which
targets vanilla OpenVLA-7b's module layout instead of OpenVLA-OFT's:

  BadVLA (OFT)  : model.vision_backbone, model.projector,
                  model.language_model.model.layers (x32 LLaMA blocks),
                  PLUS separate proprio_projector / action_head /
                  noisy_action_projector modules.
  GoBA (vanilla): model.vision_backbone, model.projector (same names),
                  model.language_model.model.layers (x32 LLaMA blocks --
                  same attribute path as BadVLA, since GoBA's eval-time model
                  is also the HF `OpenVLAForActionPrediction` wrapper, see
                  `prismatic/extern/hf/modeling_prismatic.py`). There is NO
                  separate proprio_projector / action_head /
                  noisy_action_projector in vanilla OpenVLA -- action
                  prediction is autoregressive through the LLM's own token
                  head, not a separate regression module. So only THREE
                  groups are hooked: vision, projector, llm.

NOTE on the LLM attribute path: GoBA's *training*-side model class
(`prismatic/models/backbones/llm/llama2.py`) exposes the LLaMA decoder
stack at `llm_backbone.llm.model.layers`. But the checkpoint actually
evaluated here is loaded via `experiments/robot/openvla_utils.get_vla`,
which calls `AutoModelForVision2Seq.from_pretrained(...)` and returns an
`OpenVLAForActionPrediction` instance (see
`prismatic/extern/hf/modeling_prismatic.py`, `self.language_model =
AutoModelForCausalLM.from_config(...)`). That HF wrapper exposes the LLaMA
decoder stack at `model.language_model.model.layers`, which is the path
used below. (Verified against this repo's checkpoint dir, which has
`"architectures": ["OpenVLAForActionPrediction"]` in `config.json`.)
"""

import numpy as np
import torch
import torch.nn as nn

_probe_quiet = False


def set_probe_quiet(quiet: bool) -> None:
    """When True, suppress per-forward debug noise (hooks/metrics still run)."""
    global _probe_quiet
    _probe_quiet = quiet


def _log(msg: str = "", force: bool = False) -> None:
    if _probe_quiet and not force:
        return
    print(f"[MAHA-PROBE] {msg}", flush=True)


# ======================================================================
# Hook registry container (GoBA architecture: vision / projector / llm only)
# ======================================================================

class ProbeHookGroups:
    """All hook handles + ordered layer names, grouped by model component."""

    def __init__(self):
        self.handles: list = []
        self.vision_names: list[str] = []
        self.projector_names: list[str] = []
        self.llm_names: list[str] = []

    @property
    def all_names(self) -> list[str]:
        return self.vision_names + self.projector_names + self.llm_names

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        _log(f"ProbeHookGroups.remove() -- {len(self.handles)} handles cleaned up")


# ======================================================================
# 1. Activation capture buffer + hooks
#    (copied near-verbatim from BadVLA/trial_error/paired_probe.py --
#     architecture-agnostic, no changes needed)
# ======================================================================

class Capture:
    """Collects per-layer activations for ONE forward pass."""

    def __init__(self):
        self.store = {}
        _log("Capture() created -- empty activation buffer ready")

    def reset(self):
        n_before = sum(len(v) for v in self.store.values())
        self.store = {}
        _log(f"Capture.reset() -- cleared buffer ({n_before} prior firings discarded)")

    def add(self, name, arr):
        self.store.setdefault(name, []).append(arr)

    def snapshot(self):
        snap = {k: [a.copy() for a in v] for k, v in self.store.items()}
        n_layers = len(snap)
        n_firings = sum(len(v) for v in snap.values())
        multi = {k: len(v) for k, v in snap.items() if len(v) > 1}
        _log(f"Capture.snapshot() -- saved {n_layers} layers, {n_firings} total firings")
        if multi:
            _log(f"  multi-firing layers ({len(multi)}): "
                 f"typically vision blocks (once per camera)")
            for k, cnt in list(multi.items())[:3]:
                _log(f"    {k}: {cnt} firings  raw shape={snap[k][0].shape}")
            if len(multi) > 3:
                _log(f"    ... and {len(multi) - 3} more multi-firing layers")
        return snap


def _to_numpy(out):
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not isinstance(out, torch.Tensor):
        return None
    return out.detach().float().cpu().numpy()


def _make_hook(capture, name):
    def hook(_module, _inp, out):
        arr = _to_numpy(out)
        if arr is not None:
            capture.add(name, arr)
    return hook


def _register_linears(root: nn.Module, prefix: str, capture, groups: ProbeHookGroups,
                      names_list: list[str], component: str) -> int:
    """Recursively hook every nn.Linear under root. Returns count hooked."""
    n = 0
    for subname, submod in root.named_modules():
        if not isinstance(submod, nn.Linear):
            continue
        full_name = f"{prefix}.{subname}" if subname else prefix
        groups.handles.append(submod.register_forward_hook(_make_hook(capture, full_name)))
        names_list.append(full_name)
        n += 1
        _log(f"  + HOOK  {full_name}  (Linear  in={submod.in_features}  out={submod.out_features})")
    if n == 0:
        _log(f"  NOTE: no nn.Linear found under {component} ({prefix})")
    else:
        _log(f"  => {component}: {n} Linear layers hooked")
    return n


def _register_vit_blocks(blocks, prefix: str, capture, groups: ProbeHookGroups,
                         names_list: list[str], tower: str) -> int:
    """Hook each ViT block module output (block-level, not per-Linear)."""
    n = 0
    for i, block in enumerate(blocks):
        name = f"{prefix}.block_{i:02d}"
        groups.handles.append(block.register_forward_hook(_make_hook(capture, name)))
        names_list.append(name)
        n += 1
        _log(f"  + HOOK  {name}  ({tower} block {i}, module={block.__class__.__name__})")
    if n:
        _log(f"  => {tower}: {n} ViT blocks hooked (may fire 2x per forward: 2 cameras)")
    return n


def register_mahalanobis_hooks(model, capture) -> ProbeHookGroups:
    """Register hooks on vision / projector / llm for vanilla OpenVLA-7b.

    NEW (not a port): this is GoBA's equivalent of BadVLA's
    `register_all_probe_hooks`, cut down to the three component groups that
    actually exist in vanilla OpenVLA (no proprio_projector / action_head /
    noisy_action_projector -- vanilla OpenVLA predicts actions
    autoregressively through the LLM's own token head, so those modules do
    not exist in this architecture at all and are not stubbed).

    Parameters
    ----------
    model    : the loaded `OpenVLAForActionPrediction` instance (as returned
               by `experiments/robot/robot_utils.get_model`), exposing
               `.vision_backbone`, `.projector`, `.language_model`.
    capture  : a `Capture` instance to route hook firings into.

    Returns
    -------
    ProbeHookGroups with handles and per-component name lists.
    """
    groups = ProbeHookGroups()

    _log("")
    _log("=" * 60)
    _log("register_mahalanobis_hooks: vanilla OpenVLA-7b architecture scan")
    _log("=" * 60)

    # ----- 1. Vision backbone (ViT block outputs) -----
    _log("")
    _log("[1/3] VISION BACKBONE  (model.vision_backbone)")
    vb = getattr(model, "vision_backbone", None)
    if vb is None:
        _log("  WARNING: no vision_backbone on model -- skipping vision hooks")
    else:
        _log(f"  class: {vb.__class__.__name__}")
        if hasattr(vb, "featurizer") and hasattr(vb.featurizer, "blocks"):
            _register_vit_blocks(
                vb.featurizer.blocks, "vision.featurizer",
                capture, groups, groups.vision_names, "featurizer",
            )
        else:
            _log("  WARNING: featurizer.blocks not found")
        if hasattr(vb, "fused_featurizer") and hasattr(vb.fused_featurizer, "blocks"):
            _register_vit_blocks(
                vb.fused_featurizer.blocks, "vision.fused",
                capture, groups, groups.vision_names, "fused_featurizer",
            )
        else:
            _log("  NOTE: fused_featurizer.blocks not found (single-tower checkpoint?)")

    # ----- 2. Multimodal projector (all Linear layers) -----
    _log("")
    _log("[2/3] PROJECTOR  (model.projector)")
    proj = getattr(model, "projector", None)
    if proj is None:
        _log("  WARNING: no projector on model -- skipping")
    else:
        _log(f"  class: {proj.__class__.__name__}")
        _register_linears(proj, "projector", capture, groups, groups.projector_names, "projector")

    # ----- 3. LLM decoder blocks -----
    _log("")
    _log("[3/3] LLM  (model.language_model.model.layers  x32)")
    llm_layers = model.language_model.model.layers
    for i, layer in enumerate(llm_layers):
        name = f"llm.layer_{i:02d}"
        groups.handles.append(layer.register_forward_hook(_make_hook(capture, name)))
        groups.llm_names.append(name)
        _log(f"  + HOOK  {name}  (module={layer.__class__.__name__})")

        # Also hook the attention/MLP sub-modules directly (before their
        # output is added back into the residual stream) -- same rationale
        # as BadVLA's source: hooking the full decoder-layer output lets a
        # massive-activation / attention-sink token dominate every later
        # layer's diff via the residual stream even when nothing new
        # happens to it there. Hooking the sub-module output isolates the
        # LOCAL update each layer actually contributes.
        if hasattr(layer, "self_attn"):
            attn_name = f"{name}.self_attn"
            groups.handles.append(layer.self_attn.register_forward_hook(_make_hook(capture, attn_name)))
            groups.llm_names.append(attn_name)
        if hasattr(layer, "mlp"):
            mlp_name = f"{name}.mlp"
            groups.handles.append(layer.mlp.register_forward_hook(_make_hook(capture, mlp_name)))
            groups.llm_names.append(mlp_name)
    _log(f"  => LLM: {len(groups.llm_names)} modules hooked (block + self_attn + mlp per layer)")

    _log("")
    _log("HOOK REGISTRATION COMPLETE:")
    _log(f"  vision      : {len(groups.vision_names)}")
    _log(f"  projector   : {len(groups.projector_names)}")
    _log(f"  llm         : {len(groups.llm_names)}")
    _log(f"  TOTAL hooks : {len(groups.handles)}")
    _log("=" * 60)
    return groups


# ======================================================================
# 2. Metric math (numpy only) -- copied verbatim from BadVLA's
#    trial_error/paired_probe.py. Only the Mahalanobis-relevant pieces are
#    included (pool_tokens, auroc, _split_indices,
#    compute_mahalanobis_by_group); the L2/cosine/JS drift-table functions
#    are intentionally NOT ported.
# ======================================================================

def pool_tokens(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 3:
        a = a[0]
    if a.ndim == 2:
        a = a.mean(axis=0)
    return a.reshape(-1)


# ======================================================================
# 2b. Mahalanobis detection (clean-calibrated, trigger-agnostic)
# ======================================================================
#
# Idea: model what CLEAN activations look like (per-dimension mean/std from a
# held-out calibration split of clean scenes), then score any activation by how
# far off that clean manifold it sits. This is trigger-agnostic by construction
# -- calibration never sees a trigger. Diagonal Mahalanobis:
#
#     z          = (x - mu) / sigma          (per-dimension standardization)
#     maha(x)    = || z ||_2                 (diagonal Mahalanobis distance)
#
# This is already scale-invariant to a layer's overall activation magnitude:
# mu and sigma are fit from that same layer's raw units, so rescaling a whole
# layer by any constant c rescales mu and sigma by c too and z is unchanged.
# No extra normalization (e.g. L2-normalizing x first) is needed for that --
# doing so would only discard magnitude information without adding invariance.
# The one place scale-invariance can leak in is the variance floor below,
# which is why it is defined relative to each layer's own scale rather than
# as a fixed absolute number.
#
# A clean test sample should score low; a triggered sample should score high if
# the trigger pushes activations off the clean manifold.

_GROUP_TO_NAMES_ATTR = {
    "vision": "vision_names",
    "projector": "projector_names",
    "llm": "llm_names",
}


def auroc(scores_pos, scores_neg):
    """AUROC: can scores_pos (e.g. triggered) be ranked above scores_neg (clean)?

    Same as source: ``sklearn.metrics.roc_auc_score``: label 1 = pos, 0 = neg.
    """
    from sklearn.metrics import roc_auc_score

    scores_pos = np.asarray(scores_pos, dtype=np.float64)
    scores_neg = np.asarray(scores_neg, dtype=np.float64)
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return float("nan")
    labels = np.concatenate([
        np.ones(len(scores_pos), dtype=np.int32),
        np.zeros(len(scores_neg), dtype=np.int32),
    ])
    scores = np.concatenate([scores_pos, scores_neg])
    return float(roc_auc_score(labels, scores))


def _split_indices(n, cal_fraction, seed):
    """Shuffle scene indices and split into (calibration, test)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_cal = int(round(n * cal_fraction))
    n_cal = max(1, min(n - 1, n_cal))  # keep at least 1 in each split
    return idx[:n_cal], idx[n_cal:]


def compute_mahalanobis_by_group(clean_by_layer, trig_by_layer,
                                 cal_fraction: float = 0.5, seed: int = 0,
                                 eps: float = 1e-6):
    """Per-layer diagonal Mahalanobis + a group-level detection AUROC.

    Operates on raw pooled activations (no upfront normalization). Diagonal
    z-scoring, z = (x - mu) / sigma with mu/sigma fit on clean calibration in
    that layer's own raw units, is already invariant to the layer's overall
    activation magnitude: rescaling a layer's activations by any constant c
    rescales mu and sigma by c too, so z is unchanged. See the module-level
    comment above for the proof. The only place that invariance can leak is
    the variance floor, which is therefore kept relative to the layer's own
    scale (see below) instead of a fixed absolute constant.

    Parameters
    ----------
    clean_by_layer : {layer_label: [vec_scene0, vec_scene1, ...]}  (clean run)
    trig_by_layer  : {layer_label: [vec_scene0, vec_scene1, ...]}  (triggered run)
        Both keyed identically; index = scene order.
    cal_fraction   : fraction of clean scenes used to fit mu/sigma (rest are test)
    seed           : RNG seed for the calibration/test scene split

    Returns
    -------
    dict with:
      "rows"  : [{layer, maha_clean, maha_trig, maha_delta}, ...]
      "auroc" : group-level AUROC of held-out clean (label 0) vs triggered
                (label 1) using the summed diagonal Mahalanobis across layers.
      "n_cal", "n_test"
    """
    labels = list(clean_by_layer.keys())
    if not labels:
        return {"rows": [], "auroc": float("nan"), "n_cal": 0, "n_test": 0}

    n = max(len(v) for v in clean_by_layer.values())
    if n < 2:
        return {"rows": [], "auroc": float("nan"), "n_cal": 0, "n_test": n}

    cal_idx, test_idx = _split_indices(n, cal_fraction, seed)

    rows = []
    # Accumulate summed squared-z per test scene for the group-level AUROC.
    sq_clean = np.zeros(len(test_idx), dtype=np.float64)
    sq_trig = np.zeros(len(test_idx), dtype=np.float64)

    for label in labels:
        C = np.stack(clean_by_layer[label]).astype(np.float64)  # (n, D), raw units
        T = np.stack(trig_by_layer[label]).astype(np.float64)   # (n, D), raw units
        if len(C) != n or len(T) != n:
            continue  # inconsistent firing count; skip for clean alignment

        mu = C[cal_idx].mean(axis=0)
        raw_sigma = C[cal_idx].std(axis=0)
        # Per-dimension variance floor, needed because near-constant dims have
        # raw_sigma ~ 0 and would blow z = (x-mu)/sigma up to huge values for
        # any tiny drift. The floor only ever affects dims whose real std is
        # below it; dims with genuine variance keep their own std.
        #
        # Scale the floor by the layer's OWN typical variability: the median of
        # the non-zero per-dim stds. This is data-driven for any layer that has
        # real variance somewhere, and -- critically -- scales linearly with
        # that layer's own raw magnitude, so it does not break the scale-
        # invariance of z (see module comment above): a layer with huge raw
        # activations and one with tiny raw activations each get a floor sized
        # to their own units, not a shared absolute one. Only when a layer is
        # *fully* deterministic (no dim varies -> median undefined) do we fall
        # back to the signal magnitude (RMS of the clean mean), since there is
        # then no variance to borrow a scale from. `eps` is used only as a
        # literal division-by-zero guard for the fully-zero edge case, never as
        # the dominant floor. Fit uses clean calibration only (trigger-agnostic).
        nonzero = raw_sigma[raw_sigma > eps]
        if nonzero.size > 0:
            scale = float(np.median(nonzero))
        else:
            scale = float(np.sqrt(np.mean(mu ** 2)))  # fully-deterministic fallback
        sigma_floor = max(1e-2 * scale, eps)
        sigma = np.maximum(raw_sigma, sigma_floor)

        zc = (C[test_idx] - mu) / sigma  # (n_test, D)
        zt = (T[test_idx] - mu) / sigma
        maha_clean = np.linalg.norm(zc, axis=1)  # (n_test,)
        maha_trig = np.linalg.norm(zt, axis=1)

        sq_clean += (zc ** 2).sum(axis=1)
        sq_trig += (zt ** 2).sum(axis=1)

        rows.append({
            "layer": label,
            "maha_clean": float(maha_clean.mean()),
            "maha_trig": float(maha_trig.mean()),
            "maha_delta": float(maha_trig.mean() - maha_clean.mean()),
        })

    group_auroc = auroc(np.sqrt(sq_trig), np.sqrt(sq_clean))
    return {
        "rows": rows,
        "auroc": group_auroc,
        "n_cal": len(cal_idx),
        "n_test": len(test_idx),
    }
