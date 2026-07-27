"""
experiments/robot/libero/mahalanobis_probe.py

Multi-detector backdoor-DETECTION library for GoBA_attack (vanilla
OpenVLA-7b), ported from BadVLA's `trial_error/paired_probe.py`.

Five detector types are ported here, verbatim where the math is
architecture-agnostic (everything below operates on already-pooled numpy
vectors or raw hook outputs, so none of it cares whether the model is
OpenVLA-OFT or vanilla OpenVLA):

  * L2 / relative L2      -- `l2_distance`, `relative_l2`, `compute_layer_metrics`.
  * Cosine distance        -- `cosine_distance` (raw activations, oldest code
                              in the source file) and `compute_vocab_cosine_by_group`
                              (vocab-space, via lm_head).
  * Stratified split       -- `stratified_disjoint_split`: task-stratified
                              cal/clean-test/trigger scene split. Fixes a task-
                              leakage bug that a flat "first N / last M" scene
                              cut has whenever N is a multiple of episodes-per-
                              task (see its docstring) -- GoBA's own runner
                              (`run_mahalanobis_libero_goal.py`) had exactly
                              this bug before this port (200/150/150 over 500
                              task-major scenes lands the cut precisely on
                              task boundaries: 350 = 7 x 50).
  * Mahalanobis            -- `compute_mahalanobis_by_group`, now upgraded to
                              accept optional `cal_idx`/`test_idx` (matching
                              BadVLA's current version) so it can consume the
                              stratified split above instead of only the plain
                              shuffle-split `_split_indices`.
  * Logit lens             -- `compute_logit_lens_by_group` (lm_head -> softmax
                              -> Jensen-Shannon divergence, z-scored against a
                              leave-one-out clean-calibration null).

  * `pool_tokens`                    -- pools a raw activation tensor to 1-D.
  * `_split_indices`                 -- plain (non-stratified) cal/test split.
  * `auroc`                          -- AUROC via sklearn.
  * `Capture` / `_make_hook` / `_to_numpy` / `_register_linears`
                                      -- hook-registration primitives. These
                                         are architecture-agnostic and are
                                         copied near-verbatim from the source.

All of the above are copied verbatim (or near-verbatim) from BadVLA's
`trial_error/paired_probe.py` -- do not "simplify" the variance-floor /
scale-invariance logic in the Mahalanobis and logit-lens/vocab-cosine
functions, it is load-bearing (see each function's docstring for the proof).
Two of the five (logit-lens and vocab-cosine) only existed as *uncommitted*
working-tree changes in BadVLA at port time (`git log --all -S` found them in
no commit on any branch) -- ported from the files on disk as they stood then.

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
# 2a. L2 / relative-L2 / cosine drift metrics -- copied verbatim from
#     BadVLA's trial_error/paired_probe.py. Descriptive, within-scene
#     (paired clean vs. trigger) drift -- no calibration split, no AUROC.
# ======================================================================

def l2_distance(a, b):
    return float(np.linalg.norm(a - b))


def relative_l2(clean, trig, eps: float = 1e-8):
    """Scale-invariant L2: ||trig - clean|| / ||clean||.

    Raw L2 is dominated by each layer's activation magnitude, so it cannot be
    compared across layers (e.g. llm dwarfs projector). Dividing by the clean
    activation norm makes the drift a *fraction* of the signal size, so
    values are comparable layer-to-layer regardless of scale.
    """
    denom = float(np.linalg.norm(clean)) + eps
    return float(np.linalg.norm(np.asarray(trig) - np.asarray(clean)) / denom)


def cosine_distance(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (na * nb))


def js_divergence(logits_clean, logits_triggered):
    from scipy.spatial.distance import jensenshannon
    p = _softmax(logits_clean)
    q = _softmax(logits_triggered)
    return float(jensenshannon(p, q) ** 2)


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def compute_layer_metrics(clean_store, trig_store, ordered_names, group: str = "layers",
                          quiet: bool = True):
    """Per-layer L2 / relative-L2 / cosine drift for ONE paired (clean, trig) scene."""
    if not quiet:
        _log("")
        _log(f"compute_layer_metrics({group})")
        _log(f"  layers to compare: {len(ordered_names)}")

    rows = []
    missing_clean, missing_trig, mismatched = 0, 0, 0

    for name in ordered_names:
        clean_list = clean_store.get(name, [])
        trig_list = trig_store.get(name, [])
        if not clean_list:
            missing_clean += 1
            continue
        if not trig_list:
            missing_trig += 1
            continue
        if len(clean_list) != len(trig_list):
            mismatched += 1
            if not quiet:
                _log(f"  WARNING: {name} firing mismatch "
                     f"clean={len(clean_list)} trig={len(trig_list)}")

        for occ, (hc, ht) in enumerate(zip(clean_list, trig_list)):
            pc = pool_tokens(hc)
            pt = pool_tokens(ht)
            row = {
                "layer": name,
                "l2": l2_distance(pc, pt),
                "relative_l2": relative_l2(pc, pt),
                "cosine_dist": cosine_distance(pc, pt),
            }
            if len(clean_list) > 1:
                row["occurrence"] = occ
            rows.append(row)

    if not quiet:
        if missing_clean:
            _log(f"  SKIPPED {missing_clean} layers: not in clean_store (hook never fired)")
        if missing_trig:
            _log(f"  SKIPPED {missing_trig} layers: not in trig_store")
        if mismatched:
            _log(f"  WARNING: {mismatched} layers had unequal firing counts")
        _log(f"  => computed {len(rows)} metric rows")
    return rows


def compute_action_metrics(a_clean, a_triggered, quiet: bool = True):
    """L2 / cosine distance between the predicted clean vs. triggered action."""
    ac = np.asarray(a_clean, dtype=np.float64).reshape(-1)
    at = np.asarray(a_triggered, dtype=np.float64).reshape(-1)

    metrics = {
        "l2_frobenius": l2_distance(ac, at),
        "cosine_dist": cosine_distance(ac, at),
    }
    if not quiet:
        _log(f"  => action L2: {metrics['l2_frobenius']:.6f}  "
             f"cosine: {metrics['cosine_dist']:.6f}")
    return metrics


def _aggregate_layer_metrics(rows_per_scene):
    """Mean pooled L2 / relative_l2 / cosine across scenes (BadVLA's `_aggregate`)."""
    acc, order = {}, []
    for rows in rows_per_scene:
        for r in rows:
            key = (r["layer"], r.get("occurrence"))
            if key not in acc:
                acc[key] = {"layer": r["layer"], "l2": [], "relative_l2": [], "cosine_dist": []}
                if "occurrence" in r:
                    acc[key]["occurrence"] = r["occurrence"]
                order.append(key)
            acc[key]["l2"].append(r["l2"])
            acc[key]["relative_l2"].append(r.get("relative_l2", float("nan")))
            acc[key]["cosine_dist"].append(r["cosine_dist"])
    out = []
    for key in order:
        a = acc[key]
        row = {
            "layer": a["layer"],
            "l2": float(np.mean(a["l2"])),
            "relative_l2": float(np.nanmean(a["relative_l2"])),
            "cosine_dist": float(np.mean(a["cosine_dist"])),
        }
        if "occurrence" in a:
            row["occurrence"] = a["occurrence"]
        out.append(row)
    return out


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


def stratified_disjoint_split(n_total, num_tasks, n_cal, n_clean_test, n_trig, seed):
    """Task-stratified version of the cal/clean-test/trigger scene split.

    _split_indices (and a plain-slice trigger-pool cut it's often paired with)
    only guarantees no SCENE index is reused across the three roles. Scenes
    are collected task-major (all of task 0's episodes, then all of task 1's,
    ...), so a flat "first N -> cal+clean-test, last M -> trigger" cut lands
    on a task boundary whenever N is a multiple of episodes-per-task -- which
    it always is for the 200/150/150 defaults over 10 libero_goal tasks x 50
    episodes (350 = 7 x 50). That silently makes task identity perfectly
    predictive of clean-vs-trigger (tasks 0-6 only ever clean/cal, tasks 7-9
    only ever trigger), so a detector could score well by learning "which
    task is this" instead of "was this triggered". This function fixes that:
    EVERY task contributes its own proportional share of scenes to cal,
    clean-test, AND trigger, so no role is task-specific.

    Requires n_total, n_cal, n_clean_test, and n_trig to each divide evenly
    by num_tasks. No remainder-splitting logic is implemented since nothing
    here currently needs a non-evenly-divisible split; add it if that changes.

    Returns
    -------
    (cal_idx, clean_test_idx, trig_idx) : global scene-index arrays (into the
        full 0..n_total-1 scene list, in original task-major collection
        order), each disjoint from the other two, each drawing proportionally
        from every task.
    """
    assert n_total % num_tasks == 0, (
        f"stratified_disjoint_split: n_total={n_total} must be divisible by num_tasks={num_tasks}"
    )
    episodes_per_task = n_total // num_tasks
    assert n_cal % num_tasks == 0 and n_clean_test % num_tasks == 0 and n_trig % num_tasks == 0, (
        f"stratified_disjoint_split: n_cal={n_cal}, n_clean_test={n_clean_test}, n_trig={n_trig} "
        f"must each be divisible by num_tasks={num_tasks} for an even per-task split"
    )
    per_task_cal = n_cal // num_tasks
    per_task_clean_test = n_clean_test // num_tasks
    per_task_trig = n_trig // num_tasks
    assert per_task_cal + per_task_clean_test + per_task_trig == episodes_per_task, (
        f"stratified_disjoint_split: per-task role sizes ({per_task_cal}+{per_task_clean_test}"
        f"+{per_task_trig}) must sum to episodes_per_task={episodes_per_task} -- every scene "
        f"needs exactly one role, with none left over"
    )

    rng = np.random.default_rng(seed)
    cal_idx, clean_test_idx, trig_idx = [], [], []
    for t in range(num_tasks):
        base = t * episodes_per_task
        local = rng.permutation(episodes_per_task)
        cal_idx.append(base + local[:per_task_cal])
        clean_test_idx.append(base + local[per_task_cal:per_task_cal + per_task_clean_test])
        trig_idx.append(base + local[per_task_cal + per_task_clean_test:])

    return (
        np.concatenate(cal_idx),
        np.concatenate(clean_test_idx),
        np.concatenate(trig_idx),
    )


def compute_mahalanobis_by_group(clean_by_layer, trig_by_layer,
                                 cal_fraction: float = 0.5, seed: int = 0,
                                 eps: float = 1e-6, cal_idx=None, test_idx=None):
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
        Ignored when cal_idx/test_idx are given explicitly.
    seed           : RNG seed for the calibration/test scene split
        Ignored when cal_idx/test_idx are given explicitly.
    cal_idx, test_idx : optional pre-computed scene-index arrays. Pass these
        when the caller already needs the same split elsewhere (e.g. the
        task-stratified split shared with logit-lens/vocab-cosine) -- avoids
        relying on two separate `_split_indices` calls with matching
        (n, cal_fraction, seed) reproducing the same split.

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

    if cal_idx is None or test_idx is None:
        cal_idx, test_idx = _split_indices(n, cal_fraction, seed)
    else:
        cal_idx = np.asarray(cal_idx)
        test_idx = np.asarray(test_idx)

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


def compute_logit_lens_by_group(clean_by_layer, trig_by_layer, lm_head_weight,
                                 cal_fraction: float = 0.5, seed: int = 0,
                                 eps: float = 1e-6, cal_idx=None, test_idx=None):
    """Logit-lens vocab-space detector, calibrated the same way as
    compute_mahalanobis_by_group -- same cal/test split, same "fit only on
    clean calibration scenes, score clean-test and trigger identically"
    contract -- but in the LM's OUTPUT distribution space instead of raw
    hidden activations:

      hidden state --lm_head--> logits --softmax--> token distribution
      distance = Jensen-Shannon divergence to a calibration-mean reference
      (bounded, symmetric -- unlike cosine distance on un-normalized logits)

    Each layer collapses to ONE scalar JS distance rather than a per-dim
    vector, so the analogue of Mahalanobis's per-dim z-scoring is: standardize
    that scalar against the mean/std of LEAVE-ONE-OUT clean-to-clean JS
    distances measured entirely within the calibration pool (mu_null,
    sigma_null). Because that null estimate never touches clean-test or
    trigger data, both conditions are then scored against the exact same
    full-calibration reference with the exact same z-score formula -- no
    leave-one-out-vs-full-mean asymmetry between clean and trigger.

    Per-layer z-scores are combined with a LINEAR sum across layers, not
    Mahalanobis's squared sum. That's a deliberate divergence, not an
    oversight: Mahalanobis squares-and-sums because each z there is one of D
    per-dimension coordinates whose shift direction is unknown a priori (a
    backdoor could push any raw activation dimension up or down), so
    chi-square combination is the correct way to detect "deviation in any
    direction" across those dimensions. Here there is only one feature per
    layer -- JS divergence, a >=0 "how far from typical clean" distance that
    is already one-sided by construction (bigger always means more
    anomalous). Summing standardized one-sided evidence linearly across
    layers is the standard combination for that case (a Stouffer's-method
    style combined z), and squaring it instead measurably destroys power: a
    synthetic sanity check with a consistent per-layer shift found the
    squared-sum's combined AUROC *degrading* as more (noisy) layers were
    added (0.72 @ 3 layers -> 0.57 @ 8 layers) while the linear sum saturated
    to ~1.0 by 3 layers and stayed there.

    Only "llm.layer_NN" labels (the residual-stream decoder-block output) are
    scored; "llm.layer_NN.self_attn" / ".mlp" sub-module outputs are deltas
    into the residual stream, not the residual stream itself, so they are not
    valid lm_head inputs and are skipped.

    Parameters
    ----------
    clean_by_layer, trig_by_layer, cal_fraction, seed, cal_idx, test_idx :
        same contract as compute_mahalanobis_by_group.
    lm_head_weight : the (vocab, hidden) unembedding matrix (torch tensor).
    eps : absolute division-by-zero floor for sigma_null (mirrors the
        Mahalanobis variance floor -- see that function's docstring).

    Returns
    -------
    dict with:
      "rows"  : [{layer, js_clean_mean, js_trig_mean, layer_auroc}, ...]
      "auroc" : group-level AUROC of held-out clean (label 0) vs triggered
                (label 1) using the linearly-summed per-layer z-score.
      "n_cal", "n_test"
    """
    labels = [l for l in clean_by_layer.keys()
              if l.startswith("llm.layer_") and l.count(".") == 1]
    if not labels:
        return {"rows": [], "auroc": float("nan"), "n_cal": 0, "n_test": 0}
    labels.sort(key=lambda s: int(s.rsplit("_", 1)[-1]))

    n = max(len(clean_by_layer[l]) for l in labels)
    if n < 2:
        return {"rows": [], "auroc": float("nan"), "n_cal": 0, "n_test": n}

    if cal_idx is None or test_idx is None:
        cal_idx, test_idx = _split_indices(n, cal_fraction, seed)
    else:
        cal_idx = np.asarray(cal_idx)
        test_idx = np.asarray(test_idx)
    n_cal = len(cal_idx)

    W = lm_head_weight.detach().to(dtype=torch.float32)
    device = W.device

    rows = []
    sum_clean = np.zeros(len(test_idx), dtype=np.float64)
    sum_trig = np.zeros(len(test_idx), dtype=np.float64)

    for label in labels:
        C = np.stack(clean_by_layer[label]).astype(np.float32)  # (n, D)
        T = np.stack(trig_by_layer[label]).astype(np.float32)   # (n, D)
        if len(C) != n or len(T) != n:
            continue  # inconsistent firing count; skip for clean alignment

        with torch.no_grad():
            cal_logits = (torch.tensor(C[cal_idx], device=device) @ W.T)
            clean_test_logits = (torch.tensor(C[test_idx], device=device) @ W.T)
            trig_test_logits = (torch.tensor(T[test_idx], device=device) @ W.T)
        cal_logits = cal_logits.cpu().numpy().astype(np.float64)
        clean_test_logits = clean_test_logits.cpu().numpy().astype(np.float64)
        trig_test_logits = trig_test_logits.cpu().numpy().astype(np.float64)

        cal_sum = cal_logits.sum(axis=0)
        ref_logits = cal_sum / n_cal  # full-calibration reference (scores clean-test AND trigger)

        # Leave-one-out null: how far does a typical CLEAN calibration sample
        # sit from the mean of the other calibration samples? A property of
        # the calibration pool only -- never touches clean-test or trigger
        # data, so those two conditions stay perfectly symmetric below.
        d_null = np.empty(n_cal, dtype=np.float64)
        for i in range(n_cal):
            loo_logits = (cal_sum - cal_logits[i]) / (n_cal - 1)
            d_null[i] = js_divergence(loo_logits, cal_logits[i])

        mu_null = float(d_null.mean())
        sigma_null = float(d_null.std())
        # Relative variance floor, same spirit as compute_mahalanobis_by_group:
        # guard the near-zero-spread edge case without imposing an absolute
        # scale that would break comparability across layers.
        floor = max(1e-2 * max(mu_null, eps), eps)
        sigma_null = max(sigma_null, floor)

        d_clean = np.array([js_divergence(ref_logits, l) for l in clean_test_logits])
        d_trig = np.array([js_divergence(ref_logits, l) for l in trig_test_logits])

        z_clean = (d_clean - mu_null) / sigma_null
        z_trig = (d_trig - mu_null) / sigma_null

        sum_clean += z_clean
        sum_trig += z_trig

        rows.append({
            "layer": label,
            "js_clean_mean": float(d_clean.mean()),
            "js_trig_mean": float(d_trig.mean()),
            "z_clean_mean": float(z_clean.mean()),
            "z_trig_mean": float(z_trig.mean()),
            "layer_auroc": auroc(d_trig, d_clean),
        })

    group_auroc = auroc(sum_trig, sum_clean)
    return {
        "rows": rows,
        "auroc": group_auroc,
        "n_cal": int(n_cal),
        "n_test": int(len(test_idx)),
    }


def compute_vocab_cosine_by_group(clean_by_layer, trig_by_layer, lm_head_weight,
                                  cal_fraction: float = 0.5, seed: int = 0,
                                  eps: float = 1e-6, cal_idx=None, test_idx=None):
    """Vocab-space COSINE detector.

      hidden state --lm_head--> logits
      distance = 1 - cosine_similarity(logits, calibration-mean reference)

    This differs from compute_logit_lens_by_group only in the distance: that
    one softmaxes first and uses Jensen-Shannon divergence (bounded,
    symmetric, a true metric on distributions); this one stays in raw,
    un-normalized logit space and uses cosine. Everything else -- the
    calibration reference, the leave-one-out null, the z-scoring, the linear
    cross-layer sum, the llm.layer_NN-only filter -- is deliberately
    identical, so the two AUROCs isolate the effect of the distance function
    rather than confounding it with the protocol.

    Calibration, clean-test, and trigger must come from NON-OVERLAPPING,
    task-stratified scene pools (the caller supplies cal_idx / test_idx built
    by stratified_disjoint_split). Scoring a trigger scene against its own
    paired clean twin would measure same-scene divergence, not detection, and
    would not be comparable to the Mahalanobis AUROC reported alongside it.

    Parameters / Returns
    --------------------
    Same contract as compute_logit_lens_by_group, except each row reports
    "cos_clean_mean" / "cos_trig_mean" instead of "js_clean_mean" /
    "js_trig_mean".
    """
    labels = [l for l in clean_by_layer.keys()
              if l.startswith("llm.layer_") and l.count(".") == 1]
    if not labels:
        return {"rows": [], "auroc": float("nan"), "n_cal": 0, "n_test": 0}
    labels.sort(key=lambda s: int(s.rsplit("_", 1)[-1]))

    n = max(len(clean_by_layer[l]) for l in labels)
    if n < 2:
        return {"rows": [], "auroc": float("nan"), "n_cal": 0, "n_test": n}

    if cal_idx is None or test_idx is None:
        cal_idx, test_idx = _split_indices(n, cal_fraction, seed)
    else:
        cal_idx = np.asarray(cal_idx)
        test_idx = np.asarray(test_idx)
    n_cal = len(cal_idx)

    W = lm_head_weight.detach().to(dtype=torch.float32)
    device = W.device

    rows = []
    sum_clean = np.zeros(len(test_idx), dtype=np.float64)
    sum_trig = np.zeros(len(test_idx), dtype=np.float64)

    for label in labels:
        C = np.stack(clean_by_layer[label]).astype(np.float32)  # (n, D)
        T = np.stack(trig_by_layer[label]).astype(np.float32)   # (n, D)
        if len(C) != n or len(T) != n:
            continue  # inconsistent firing count; skip for clean alignment

        with torch.no_grad():
            cal_logits = (torch.tensor(C[cal_idx], device=device) @ W.T)
            clean_test_logits = (torch.tensor(C[test_idx], device=device) @ W.T)
            trig_test_logits = (torch.tensor(T[test_idx], device=device) @ W.T)
        cal_logits = cal_logits.cpu().numpy().astype(np.float64)
        clean_test_logits = clean_test_logits.cpu().numpy().astype(np.float64)
        trig_test_logits = trig_test_logits.cpu().numpy().astype(np.float64)

        cal_sum = cal_logits.sum(axis=0)
        ref_logits = cal_sum / n_cal  # full-calibration reference (scores clean-test AND trigger)

        # Leave-one-out null within the calibration pool only -- never touches
        # clean-test or trigger, so both stay symmetric below (same reasoning
        # as compute_logit_lens_by_group).
        d_null = np.empty(n_cal, dtype=np.float64)
        for i in range(n_cal):
            loo_logits = (cal_sum - cal_logits[i]) / (n_cal - 1)
            d_null[i] = cosine_distance(loo_logits, cal_logits[i])

        mu_null = float(d_null.mean())
        sigma_null = float(d_null.std())
        floor = max(1e-2 * max(mu_null, eps), eps)
        sigma_null = max(sigma_null, floor)

        d_clean = np.array([cosine_distance(ref_logits, l) for l in clean_test_logits])
        d_trig = np.array([cosine_distance(ref_logits, l) for l in trig_test_logits])

        z_clean = (d_clean - mu_null) / sigma_null
        z_trig = (d_trig - mu_null) / sigma_null

        sum_clean += z_clean
        sum_trig += z_trig

        rows.append({
            "layer": label,
            "cos_clean_mean": float(d_clean.mean()),
            "cos_trig_mean": float(d_trig.mean()),
            "z_clean_mean": float(z_clean.mean()),
            "z_trig_mean": float(z_trig.mean()),
            "layer_auroc": auroc(d_trig, d_clean),
        })

    group_auroc = auroc(sum_trig, sum_clean)
    return {
        "rows": rows,
        "auroc": group_auroc,
        "n_cal": int(n_cal),
        "n_test": int(len(test_idx)),
    }


# ======================================================================
# 3. Reporting -- copied verbatim from BadVLA's trial_error/paired_probe.py,
#    minus the proprio/action_head/noisy_action sections (those groups don't
#    exist for vanilla OpenVLA -- see the module docstring).
# ======================================================================

def _table_row_label(row):
    """Format a table row's layer name, with an occurrence suffix if the hook fired
    more than once per scene.

    The occurrence's meaning depends on WHY the hook fired more than once, which
    differs by group: vision/projector hooks only ever re-fire because a single
    forward pass processes more than one camera image (BadVLA's OFT model predicts
    the whole action chunk in one forward pass, so any repeat firing there is
    per-camera). LLM decoder-layer hooks fire once per `generate()` step instead --
    vanilla OpenVLA decodes autoregressively (`max_new_tokens=action_dim`), so all
    32 decoder layers run once per generated action token, unrelated to camera
    count (this repo's observation only ever has 1 camera anyway). So the suffix
    is `#stepN` for llm.* rows and `#camN` everywhere else.
    """
    label = row["layer"]
    if label.startswith("llm.layer_"):
        rest = label[len("llm.layer_"):]
        num, _, suffix = rest.partition(".")
        try:
            short = f"{int(num):>2}"
            label = f"{short}.{suffix}" if suffix else short
        except ValueError:
            pass
    if "occurrence" in row:
        tag = "step" if row["layer"].startswith("llm.") else "cam"
        label = f"{label}#{tag}{row['occurrence']}"
    return label


def _table_fmt_float(value):
    if value != value:  # NaN
        return "N/A"
    return f"{value:.4f}"


def _format_table(rows, title):
    """Layer drift table: pooled L2 / Rel L2 / cosine."""
    col_layer, col_l2, col_rel, col_cos = "Layer", "L2 (pooled)", "Rel L2", "Cosine"
    labels = [_table_row_label(r) for r in rows]
    l2_vals = [_table_fmt_float(r["l2"]) for r in rows]
    rel_vals = [_table_fmt_float(r.get("relative_l2", float("nan"))) for r in rows]
    cos_vals = [_table_fmt_float(r["cosine_dist"]) for r in rows]

    w_layer = max(len(col_layer), max((len(l) for l in labels), default=0))
    w_l2 = max(len(col_l2), max((len(v) for v in l2_vals), default=0))
    w_rel = max(len(col_rel), max((len(v) for v in rel_vals), default=0))
    w_cos = max(len(col_cos), max((len(v) for v in cos_vals), default=0))

    sep = f"{'-' * w_layer}-+-{'-' * w_l2}-+-{'-' * w_rel}-+-{'-' * w_cos}"
    lines = [
        title,
        (f"{col_layer:<{w_layer}} | {col_l2:>{w_l2}} | {col_rel:>{w_rel}} | "
         f"{col_cos:>{w_cos}}"),
        sep,
    ]
    for label, l2, rel, cos in zip(labels, l2_vals, rel_vals, cos_vals):
        lines.append(f"{label:<{w_layer}} | {l2:>{w_l2}} | {rel:>{w_rel}} | {cos:>{w_cos}}")
    return lines


def _format_maha_table(maha_result, title):
    """Format a Mahalanobis result dict as a 4-column ASCII table."""
    rows = maha_result.get("rows", [])
    if not rows:
        return [title, "  (no data)"]

    col_layer, col_clean, col_trig, col_delta = "Layer", "Maha(clean)", "Maha(trig)", "Delta"
    labels = [r["layer"] for r in rows]
    clean_vals = [_table_fmt_float(r["maha_clean"]) for r in rows]
    trig_vals = [_table_fmt_float(r["maha_trig"]) for r in rows]
    delta_vals = [_table_fmt_float(r["maha_delta"]) for r in rows]

    w_layer = max(len(col_layer), max((len(l) for l in labels), default=0))
    w_cl = max(len(col_clean), max((len(v) for v in clean_vals), default=0))
    w_tr = max(len(col_trig), max((len(v) for v in trig_vals), default=0))
    w_de = max(len(col_delta), max((len(v) for v in delta_vals), default=0))

    sep = f"{'-'*w_layer}-+-{'-'*w_cl}-+-{'-'*w_tr}-+-{'-'*w_de}"
    lines = [
        title,
        f"{col_layer:<{w_layer}} | {col_clean:>{w_cl}} | {col_trig:>{w_tr}} | {col_delta:>{w_de}}",
        sep,
    ]
    for lab, cl, tr, de in zip(labels, clean_vals, trig_vals, delta_vals):
        lines.append(f"{lab:<{w_layer}} | {cl:>{w_cl}} | {tr:>{w_tr}} | {de:>{w_de}}")

    n_cal, n_test = maha_result.get("n_cal", "?"), maha_result.get("n_test", "?")
    auc = maha_result.get("auroc", float("nan"))
    lines.append(f"  cal={n_cal} test={n_test}  group-AUROC={_table_fmt_float(auc)}")
    return lines


def _format_vocab_detector_table(result, title, dist_name, clean_key, trig_key):
    """Shared ASCII table for the two vocab-space detectors (logit-lens JS and
    vocab-cosine): identical row schema apart from the distance name, so they
    share one layout -- keeping outputs directly comparable.
    """
    rows = result.get("rows", [])
    if not rows:
        return [title, "  (no data)"]

    col_layer, col_auc = "Layer", "Layer AUROC"
    col_jsc, col_jst = f"{dist_name}(clean)", f"{dist_name}(trig)"
    col_zc, col_zt = "z(clean)", "z(trig)"

    labels = [r["layer"] for r in rows]
    auc_vals = [_table_fmt_float(r["layer_auroc"]) for r in rows]
    jsc_vals = [_table_fmt_float(r[clean_key]) for r in rows]
    jst_vals = [_table_fmt_float(r[trig_key]) for r in rows]
    zc_vals = [_table_fmt_float(r["z_clean_mean"]) for r in rows]
    zt_vals = [_table_fmt_float(r["z_trig_mean"]) for r in rows]

    w_layer = max(len(col_layer), max((len(l) for l in labels), default=0))
    w_auc = max(len(col_auc), max((len(v) for v in auc_vals), default=0))
    w_jsc = max(len(col_jsc), max((len(v) for v in jsc_vals), default=0))
    w_jst = max(len(col_jst), max((len(v) for v in jst_vals), default=0))
    w_zc = max(len(col_zc), max((len(v) for v in zc_vals), default=0))
    w_zt = max(len(col_zt), max((len(v) for v in zt_vals), default=0))

    sep = f"{'-'*w_layer}-+-{'-'*w_auc}-+-{'-'*w_jsc}-+-{'-'*w_jst}-+-{'-'*w_zc}-+-{'-'*w_zt}"
    lines = [
        title,
        (f"{col_layer:<{w_layer}} | {col_auc:>{w_auc}} | {col_jsc:>{w_jsc}} | "
         f"{col_jst:>{w_jst}} | {col_zc:>{w_zc}} | {col_zt:>{w_zt}}"),
        sep,
    ]
    for lab, ac, jc, jt, zc, zt in zip(labels, auc_vals, jsc_vals, jst_vals, zc_vals, zt_vals):
        lines.append(f"{lab:<{w_layer}} | {ac:>{w_auc}} | {jc:>{w_jsc}} | {jt:>{w_jst}} | {zc:>{w_zc}} | {zt:>{w_zt}}")

    n_cal, n_test = result.get("n_cal", "?"), result.get("n_test", "?")
    auc = result.get("auroc", float("nan"))
    lines.append(f"  cal={n_cal} test={n_test}  group-AUROC={_table_fmt_float(auc)}  (linear sum of per-layer z-scores)")
    return lines


def _format_logit_lens_table(ll_result, title):
    return _format_vocab_detector_table(ll_result, title, "JS", "js_clean_mean", "js_trig_mean")


def _format_vocab_cosine_table(vc_result, title):
    return _format_vocab_detector_table(vc_result, title, "cos", "cos_clean_mean", "cos_trig_mean")


def format_summary_section(metrics: dict):
    """Format one metrics bundle: L2/RelL2/cosine drift tables, then
    Mahalanobis / logit-lens / vocab-cosine AUROC sections when present.
    GoBA only has vision/projector/llm groups (no proprio/action_head/
    noisy_action -- vanilla OpenVLA has no separate regression head).
    """
    lines = []
    if metrics.get("llm"):
        lines.extend(_format_table(
            metrics["llm"],
            "LLM decoder blocks (#step0 = full-sequence mean at the prefill forward pass; "
            "#step1+ = single newly-generated token per subsequent autoregressive decode step):",
        ))
    act = metrics.get("action")
    if act:
        lines.append("")
        lines.append(f"Action L2 (Frobenius): {act.get('l2_frobenius', 0):.4f}")
        lines.append(f"Action cosine distance: {act.get('cosine_dist', 0):.4f}")

    for key, title in (
        ("projector", "Projector layers:"),
        ("vision", "Vision ViT blocks:"),
    ):
        if metrics.get(key):
            lines.append("")
            lines.extend(_format_table(metrics[key], title))

    if metrics.get("mahalanobis"):
        lines.append("")
        lines.append("=== Mahalanobis Detection (diagonal, clean-calibrated) ===")
        lines.append("  Drift tables above: pooled L2 / Rel L2 / cosine (mean over tokens).")
        lines.append("  Mahalanobis uses pooled, clean-calibrated z-scores for detection AUROC.")
        lines.append("  AUROC: P(Maha(trig) > Maha(clean)) on held-out test scenes.")
        for grp_key, grp_title in (
            ("llm", "LLM decoder blocks:"),
            ("projector", "Projector layers:"),
            ("vision", "Vision ViT blocks:"),
        ):
            maha = metrics["mahalanobis"].get(grp_key)
            if maha and maha.get("rows"):
                lines.append("")
                lines.extend(_format_maha_table(maha, grp_title))
        aurocs = {
            g: metrics["mahalanobis"][g]["auroc"]
            for g in metrics["mahalanobis"]
            if metrics["mahalanobis"].get(g)
            and metrics["mahalanobis"][g].get("auroc") == metrics["mahalanobis"][g].get("auroc")
        }
        if aurocs:
            lines.append("")
            lines.append("  Group-level detection AUROC summary:")
            for g, auc in aurocs.items():
                lines.append(f"    {g:<15}: {_table_fmt_float(auc)}")

    if metrics.get("logit_lens"):
        lines.append("")
        lines.append("=== Logit-Lens Detection (softmax + Jensen-Shannon, clean-calibrated) ===")
        lines.append("  hidden state -> lm_head -> softmax -> token distribution -> JS divergence")
        lines.append("  vs. a full-calibration reference distribution, z-scored per layer against")
        lines.append("  leave-one-out clean-to-clean JS spread within calibration.")
        ll_llm = metrics["logit_lens"].get("llm")
        if ll_llm and ll_llm.get("rows"):
            lines.append("")
            lines.extend(_format_logit_lens_table(ll_llm, "LLM decoder blocks (logit lens):"))

    if metrics.get("vocab_cosine"):
        lines.append("")
        lines.append("=== Vocab-Cosine Detection (raw logits + cosine, clean-calibrated) ===")
        lines.append("  hidden state -> lm_head -> logit vector -> 1 - cosine_similarity vs. a")
        lines.append("  full-calibration reference logit vector, z-scored per layer against")
        lines.append("  leave-one-out clean-to-clean cosine spread within calibration (identical")
        lines.append("  protocol and identical scene split to Mahalanobis and logit-lens above).")
        vc_llm = metrics["vocab_cosine"].get("llm")
        if vc_llm and vc_llm.get("rows"):
            lines.append("")
            lines.extend(_format_vocab_cosine_table(vc_llm, "LLM decoder blocks (vocab cosine):"))

    return "\n".join(lines)


def results_header(cfg_summary: dict, maha_by_group: dict,
                   logit_lens_by_group: dict, vocab_cosine_by_group: dict) -> str:
    """Headline AUROC block prepended to the summary table (adapted from
    BadVLA's `_results_header`). `cfg_summary` is a plain dict (not the
    draccus dataclass) so this stays independent of the runner's config type.
    """
    ll_auc = (logit_lens_by_group or {}).get("llm", {}).get("auroc", float("nan"))
    vc_auc = (vocab_cosine_by_group or {}).get("llm", {}).get("auroc", float("nan"))
    maha_llm = (maha_by_group or {}).get("llm", {}).get("auroc", float("nan"))

    lines = [
        "=" * 78,
        "GoBA backdoor probe -- clean vs. physical-trigger scenes (stratified split)",
        "=" * 78,
        f"checkpoint:  {cfg_summary.get('checkpoint')}",
        f"task suite:  {cfg_summary.get('task_suite_name')}  (trigger={cfg_summary.get('trigger_obj', 'poison_1')})",
        f"split:       cal={cfg_summary.get('n_cal')}  clean-test={cfg_summary.get('n_clean_test')}  "
        f"trigger={cfg_summary.get('n_trig')}  (disjoint pools, task-stratified)",
        "             All detectors share this split, the clean-only calibration, and the",
        "             'score clean-test and trigger identically' contract, so their AUROCs",
        "             are directly comparable -- only the distance function differs.",
        "",
        "HEADLINE DETECTION AUROC (llm group, all layers combined)",
        f"  [1] mahalanobis  = {maha_llm:.4f}   raw activations, per-dim z, squared sum",
        f"  [2] logit lens   = {ll_auc:.4f}   lm_head -> softmax -> JS, linear z-sum",
        f"  [3] vocab cosine = {vc_auc:.4f}   lm_head -> cosine on raw logits, linear z-sum",
    ]
    if maha_by_group:
        lines.append("")
        lines.append("Mahalanobis AUROC by group (only detector that covers non-llm groups):")
        for grp, res in maha_by_group.items():
            lines.append(f"  {grp:<15}: {res.get('auroc', float('nan')):.4f}")
    lines += [
        "",
        "L2 / relative-L2 / cosine drift tables are descriptive (paired same-scene",
        "clean-vs-trigger comparison, no calibration split, no AUROC) -- see below.",
        "",
        "Per-layer tables for every detector and every group follow below.",
        "=" * 78,
        "",
    ]
    return "\n".join(lines) + "\n"
