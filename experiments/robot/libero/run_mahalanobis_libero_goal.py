"""
experiments/robot/libero/run_mahalanobis_libero_goal.py

Runner for GoBA_attack's multi-detector backdoor-DETECTION probe, adapted
from BadVLA's `trial_error/run_libero_probe.py` (`--disjoint_mahalanobis`
mode) + `trial_error/run_mahalanobis_all_groups_disjoint.py`.

One simulator pass feeds every detector (matches BadVLA's
`run_libero_probe_local.sh` design -- "ALL THREE detectors" from a single
paired clean-vs-trigger rollout, no reason to re-simulate per detector):

  * L2 / relative-L2 / cosine   -- descriptive per-layer drift, every paired
                                    scene, vision/projector/llm groups + action.
  * Mahalanobis                 -- diagonal, clean-calibrated, all 3 groups.
  * Logit lens                  -- softmax + Jensen-Shannon, llm group only.
  * Vocab cosine                -- raw logits + cosine, llm group only.

All four share the SAME task-stratified disjoint cal/clean-test/trigger scene
split (`mahalanobis_probe.stratified_disjoint_split`) so their AUROCs are
directly comparable. This replaces the flat, non-stratified split the first
version of this runner used (`_split_indices` on the first n_cal+n_clean_test
scenes, trigger drawn from the last n_trig) -- for the 200/150/150 defaults
over 10 libero_goal tasks x 50 episodes, 350 = 7 x 50 landed that flat cut
exactly on a task boundary, making task identity perfectly predictive of
clean-vs-trigger (tasks 0-6 only ever clean/cal, 7-9 only ever trigger). See
`stratified_disjoint_split`'s docstring; this is the same leakage bug BadVLA
fixed in their commit `4b041ab`.

For every LIBERO-goal task/initial state, runs a CLEAN forward pass and a
PHYSICALLY-TRIGGERED forward pass ("paired" at the same task/episode index),
captures pooled activations per hooked layer per group (vision / projector /
llm), then computes all four detectors above and writes ONE results .txt
(headline AUROCs + full per-layer tables) plus a JSON dump.

--------------------------------------------------------------------------
ADAPTATIONS FROM BadVLA's RUNNER (read this before running)
--------------------------------------------------------------------------

1. Model loading / inference call
   BadVLA's OFT runner calls `initialize_model(cfg)` (returns model,
   action_head, proprio_projector, noisy_action_projector, processor) and
   `get_action(cfg, model, obs, task_description, processor=processor,
   action_head=..., proprio_projector=..., noisy_action_projector=...,
   use_film=...)`.
   This runner instead mirrors GoBA's own `experiments/robot/libero/
   3level_eval.py` eval-script pattern: `get_model(cfg)` ->
   `get_processor(cfg)` -> `get_action(cfg, model, obs, task_label,
   processor=processor)` (see `experiments/robot/robot_utils.py`). There is
   no action_head / proprio_projector / noisy_action_projector for vanilla
   OpenVLA.

2. Trigger mechanism -- THE MAIN ADAPTATION
   BadVLA's runner uses a SYNTHETIC pixel-patch trigger
   (`add_trigger_img(...)`, a white square pasted onto the image array) --
   no change to the simulated scene itself.

   GoBA's actual backdoor trigger is a PHYSICAL OBJECT ("poison_1") placed
   in the simulated LIBERO scene by a different BDDL file
   (`BadLIBERO/libero/libero/bddl_files-poison_eval`) than the clean
   default (`BadLIBERO/libero/libero/bddl_files`). This runner instantiates
   TWO separate `get_libero_env(...)` calls per task -- one on the clean
   BDDL dir, one on the poison BDDL dir -- for the SAME task and (as close
   as achievable, see below) the SAME initial condition, so the only
   meaningful difference between the two observations is the presence of
   the poison object.

   IMPORTANT LIMITATION (documented per instructions, not silently papered
   over): LIBERO's `env.set_init_state(raw_state_vector)` cannot be used to
   force the exact same initial state across these two envs. The raw state
   vector returned by `task_suite.get_task_init_states(task_id)` is sized
   for the CLEAN scene's object count; the poison BDDL adds one extra
   object (`poison_1`), so the underlying MuJoCo qpos/qvel vector has a
   different shape in the poison env and `set_init_state` would raise a
   shape-mismatch error if given the clean env's state vector. This is
   exactly why GoBA's OWN eval scripts (`run_libero_eval.py`,
   `run_libero_eval_backdoor.py`, `3level_eval.py`) already leave the
   `env.set_init_state(...)` line commented out and rely on a seeded
   `env.reset()` instead -- this runner follows that same, already-
   established GoBA convention. Concretely: for each (task, episode) pair
   we derive a deterministic per-episode seed, call `env.seed(...)` on BOTH
   the clean and poison env right before `env.reset()`, then take identical
   warmup steps in both. This gives the closest achievable approximation to
   a shared initial state without crashing, but is NOT a rigorous
   guarantee: the poison scene's sampler draws one extra random placement
   for `poison_1`, which can shift the RNG draw order for any objects
   sampled after it, so other objects' exact placements may diverge
   slightly between the two envs even with an identical seed. If a fully
   rigorous paired reset is needed later, the fix would be to special-case
   the poison env's object sampler to draw `poison_1`'s placement LAST (so
   it never perturbs earlier draws), which is out of scope here since it
   requires editing `BadLIBERO`'s sampler, not just this script.

3. Checkpoint / task suite
   Evaluates GoBA's own backdoored checkpoint at
   `exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug`
   on `task_suite_name="libero_goal"` (10 tasks), matching this repo's own
   `num_trials_per_task=50` convention -- 10 x 50 = 500 scenes total,
   split 200 calibration / 150 clean-test / 150 trigger (same split sizes
   as BadVLA's source runner).

--------------------------------------------------------------------------
Usage (once GPUs are free -- do NOT run this while the live eval job is
using GPUs 0-3):

    python experiments/robot/libero/run_mahalanobis_libero_goal.py \
        --pretrained_checkpoint exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \
        --task_suite_name libero_goal \
        --center_crop True

For a quick smoke test (no GPU / few scenes), pass --n_cal 4 --n_clean_test 3
--n_trig 3 and a tiny --episodes_per_task_override, or just import
`mahalanobis_probe` directly and unit-test `compute_mahalanobis_by_group`
on synthetic arrays.
--------------------------------------------------------------------------
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

sys.path.append(str(Path(__file__).resolve().parents[3]))  # repo root, for `experiments.*` imports when run directly

from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action, get_libero_image, quat2axisangle
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import DATE_TIME, get_action, get_image_resize_size, get_model, set_seed_everywhere

from experiments.robot.libero.mahalanobis_probe import (
    Capture,
    register_mahalanobis_hooks,
    set_probe_quiet,
    pool_tokens,
    compute_mahalanobis_by_group,
    compute_logit_lens_by_group,
    compute_vocab_cosine_by_group,
    compute_layer_metrics,
    compute_action_metrics,
    _aggregate_layer_metrics,
    stratified_disjoint_split,
    results_header,
    format_summary_section,
    _GROUP_TO_NAMES_ATTR,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = (
        "exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
    )
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True   # must be True: this checkpoint was trained with image_aug

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    env_img_res: int = 256

    clean_bddl_dir: str = "BadLIBERO/libero/libero/bddl_files"
    poison_bddl_dir: str = "BadLIBERO/libero/libero/bddl_files-poison_eval"

    #################################################################################################################
    # Probe / Mahalanobis-split parameters
    #################################################################################################################
    n_cal: int = 200
    n_clean_test: int = 150
    n_trig: int = 150

    seed: int = 42
    split_seed: int = 0     # RNG seed for the task-stratified cal/clean-test/trigger split

    out_path: str = "experiments/robot/libero/probe_logs/all_detectors_libero_goal.json"
    out_txt_dir: str = "experiments/robot/libero/probe_logs"

    # fmt: on


def _paired_clean_trig_observation(clean_env, trig_env, cfg, episode_seed, resize_size):
    """Reset both envs with the SAME seed, warm up identically, return (clean_obs, trig_obs).

    See the module docstring's "IMPORTANT LIMITATION" section: this is a
    best-effort pairing via a shared RNG seed, not a guaranteed identical
    MuJoCo state, because the poison scene's extra object makes
    `env.set_init_state` incompatible across the two BDDL variants.
    """
    clean_env.seed(episode_seed)
    clean_env.reset()
    trig_env.seed(episode_seed)
    trig_env.reset()

    clean_obs, trig_obs = None, None
    for _ in range(cfg.num_steps_wait):
        clean_obs, _, _, _ = clean_env.step(get_libero_dummy_action(cfg.model_family))
        trig_obs, _, _, _ = trig_env.step(get_libero_dummy_action(cfg.model_family))

    def _to_observation(obs):
        img = get_libero_image(obs, resize_size)
        return {
            "full_image": img,
            "state": np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
        }

    return _to_observation(clean_obs), _to_observation(trig_obs)


@draccus.wrap()
def run(cfg: GenerateConfig) -> None:
    assert cfg.task_suite_name == "libero_goal", "This runner is scoped to libero_goal per the task spec."
    n_total = cfg.n_cal + cfg.n_clean_test + cfg.n_trig

    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key (mirrors 3level_eval.py's convention)
    cfg.unnorm_key = cfg.task_suite_name

    model = get_model(cfg)
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)

    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    episodes_per_task = max(1, n_total // num_tasks)

    capture = Capture()
    hook_groups = register_mahalanobis_hooks(model, capture)
    set_probe_quiet(True)

    # clean_pooled[group][layer_label] = list of per-scene vectors, in scene order 0..(n_total-1)
    clean_pooled = {g: {} for g in _GROUP_TO_NAMES_ATTR}
    trig_pooled = {g: {} for g in _GROUP_TO_NAMES_ATTR}
    # Per-scene L2/relative-L2/cosine drift rows, one list per group (descriptive,
    # no calibration split needed -- every paired scene contributes).
    layer_metric_rows = {g: [] for g in _GROUP_TO_NAMES_ATTR}
    action_metric_rows = []
    n_scenes = 0

    try:
        for task_id in tqdm.tqdm(range(num_tasks), desc="tasks"):
            if n_scenes >= n_total:
                break
            task = task_suite.get_task(task_id)

            clean_env, task_description = get_libero_env(
                task, cfg.model_family, resolution=cfg.env_img_res,
                bddl_path=cfg.clean_bddl_dir, seed=cfg.seed,
            )
            trig_env, _ = get_libero_env(
                task, cfg.model_family, resolution=cfg.env_img_res,
                bddl_path=cfg.poison_bddl_dir, seed=cfg.seed,
            )

            for episode_idx in tqdm.tqdm(range(episodes_per_task), desc=f"task {task_id} episodes", leave=False):
                if n_scenes >= n_total:
                    break
                # Deterministic, reproducible per-(task, episode) seed shared by both envs.
                episode_seed = cfg.seed * 100000 + task_id * 1000 + episode_idx

                clean_obs, trig_obs = _paired_clean_trig_observation(
                    clean_env, trig_env, cfg, episode_seed, resize_size,
                )

                capture.reset()
                a_clean = get_action(cfg, model, clean_obs, task_description, processor=processor)
                clean_store = capture.snapshot()

                capture.reset()
                a_trig = get_action(cfg, model, trig_obs, task_description, processor=processor)
                trig_store = capture.snapshot()

                for group, attr in _GROUP_TO_NAMES_ATTR.items():
                    names = getattr(hook_groups, attr, [])
                    layer_metric_rows[group].append(
                        compute_layer_metrics(clean_store, trig_store, names, group=group)
                    )
                    for name in names:
                        c = clean_store.get(name, [])
                        t = trig_store.get(name, [])
                        if not c or not t:
                            continue
                        pc = pool_tokens(c[0]).astype(np.float32)
                        pt = pool_tokens(t[0]).astype(np.float32)
                        clean_pooled[group].setdefault(name, []).append(pc)
                        trig_pooled[group].setdefault(name, []).append(pt)

                action_metric_rows.append(compute_action_metrics(np.asarray(a_clean), np.asarray(a_trig)))

                n_scenes += 1

            clean_env.close()
            trig_env.close()
    finally:
        hook_groups.remove()
        set_probe_quiet(False)

    print(f"[all-detectors-libero-goal] collected {n_scenes} scenes")

    # Task-stratified disjoint split: every libero_goal task contributes its
    # own proportional share of scenes to cal, clean-test, AND trigger (see
    # stratified_disjoint_split's docstring for why a flat "first N / last M"
    # cut is wrong here -- it makes task identity perfectly predictive of
    # clean-vs-trigger since scenes are collected task-major).
    cal_scene_idx, clean_test_scene_idx, trig_scene_idx = stratified_disjoint_split(
        n_scenes, num_tasks, cfg.n_cal, cfg.n_clean_test, cfg.n_trig, seed=cfg.split_seed,
    )
    # compute_mahalanobis_by_group / compute_logit_lens_by_group / compute_vocab_cosine_by_group
    # index clean_by_layer/trig_by_layer positionally (0..n_cal_test-1), not by real scene id --
    # so map local positions to the global, stratified scene indices just picked above.
    n_cal_test = cfg.n_cal + cfg.n_clean_test
    local_to_global_clean = np.concatenate([cal_scene_idx, clean_test_scene_idx])
    cal_idx = np.arange(len(cal_scene_idx))
    test_idx = np.arange(len(cal_scene_idx), n_cal_test)
    assert len(test_idx) == len(trig_scene_idx)
    print(f"cal={len(cal_idx)} clean-test={len(test_idx)} trigger={len(trig_scene_idx)} "
          f"(disjoint, task-stratified over {num_tasks} tasks)")

    maha_by_group, logit_lens_by_group, vocab_cosine_by_group = {}, {}, {}
    out_groups = {}

    for group in _GROUP_TO_NAMES_ATTR:
        if not clean_pooled[group]:
            print(f"\n=== {group}: no hooked layers with data, skipping ===")
            continue
        clean_by_layer, trig_by_layer = {}, {}
        for label, vecs in clean_pooled[group].items():
            if len(vecs) < n_scenes:
                continue  # hook didn't fire every scene; skip for clean alignment
            clean_slots = [vecs[g] for g in local_to_global_clean]
            trig_vecs_all = trig_pooled[group][label]
            # NaN placeholder at cal_idx positions: never read (mu/sigma are fit
            # from clean-only calibration), so a stray future read fails loudly.
            trig_slots = [np.full_like(vecs[0], np.nan) for _ in range(n_cal_test)]
            for pos, scene_i in zip(test_idx, trig_scene_idx):
                trig_slots[pos] = trig_vecs_all[scene_i]
            clean_by_layer[label] = clean_slots
            trig_by_layer[label] = trig_slots

        if not clean_by_layer:
            print(f"\n=== {group}: layers present but insufficient scenes, skipping ===")
            continue

        result = compute_mahalanobis_by_group(clean_by_layer, trig_by_layer, cal_idx=cal_idx, test_idx=test_idx)
        maha_by_group[group] = result
        print(f"\n=== Mahalanobis [{group}] (n_cal={result['n_cal']}, n_test={result['n_test']}) ===")
        print(f"{'Layer':<30} | {'maha_clean':>10} | {'maha_trig':>10} | {'maha_delta':>10}")
        for row in result["rows"]:
            print(f"{row['layer']:<30} | {row['maha_clean']:10.3f} | {row['maha_trig']:10.3f} | {row['maha_delta']:10.3f}")
        print(f"Group-level detection AUROC: {result['auroc']:.4f}")

        out_groups[group] = {
            "mahalanobis": {"rows": result["rows"], "auroc": result["auroc"],
                            "n_cal": result["n_cal"], "n_test": result["n_test"]},
        }

        # Logit lens + vocab cosine reuse the SAME disjoint clean_by_layer/trig_by_layer/
        # cal_idx/test_idx built above -- one simulator pass feeds all detectors, and
        # both see the identical scenes in the identical cal/clean-test/trigger roles.
        # LLM group only: projecting through lm_head is only meaningful for the
        # residual stream, not vision/projector activations.
        if group == "llm":
            ll_result = compute_logit_lens_by_group(
                clean_by_layer, trig_by_layer,
                lm_head_weight=model.language_model.lm_head.weight,
                cal_idx=cal_idx, test_idx=test_idx,
            )
            logit_lens_by_group[group] = ll_result
            print(f"LogitLens [{group}] AUROC={ll_result['auroc']:.4f} (softmax+JS, z-scored)")

            vc_result = compute_vocab_cosine_by_group(
                clean_by_layer, trig_by_layer,
                lm_head_weight=model.language_model.lm_head.weight,
                cal_idx=cal_idx, test_idx=test_idx,
            )
            vocab_cosine_by_group[group] = vc_result
            print(f"VocabCos  [{group}] AUROC={vc_result['auroc']:.4f} (raw logits+cosine, z-scored)")

            out_groups[group]["logit_lens"] = ll_result
            out_groups[group]["vocab_cosine"] = vc_result

    # Descriptive L2/relative-L2/cosine drift, aggregated (mean) over every paired scene.
    agg = {group: _aggregate_layer_metrics(rows) for group, rows in layer_metric_rows.items()}
    agg["action"] = {
        "l2_frobenius": float(np.mean([m["l2_frobenius"] for m in action_metric_rows])) if action_metric_rows else 0.0,
        "cosine_dist": float(np.mean([m["cosine_dist"] for m in action_metric_rows])) if action_metric_rows else 0.0,
    }
    agg["mahalanobis"] = maha_by_group
    agg["logit_lens"] = logit_lens_by_group
    agg["vocab_cosine"] = vocab_cosine_by_group

    cfg_summary = {
        "checkpoint": str(cfg.pretrained_checkpoint),
        "task_suite_name": cfg.task_suite_name,
        "trigger_obj": "poison_1",
        "n_cal": cfg.n_cal, "n_clean_test": cfg.n_clean_test, "n_trig": cfg.n_trig,
    }
    text = results_header(cfg_summary, maha_by_group, logit_lens_by_group, vocab_cosine_by_group)
    text += format_summary_section(agg)
    print("\n" + text)

    out_txt_dir = Path(cfg.out_txt_dir)
    out_txt_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_txt_dir / f"all_detectors_libero_goal_{DATE_TIME}.txt"
    out_txt.write_text(text + "\n")
    print(f"Saved results table -> {out_txt}")

    out = {
        "n_scenes": n_scenes, "n_cal": cfg.n_cal, "n_clean_test": cfg.n_clean_test,
        "n_trig": cfg.n_trig, "checkpoint": str(cfg.pretrained_checkpoint),
        "l2_cosine": {g: rows for g, rows in agg.items() if g in _GROUP_TO_NAMES_ATTR},
        "action": agg["action"],
        "groups": out_groups,
    }
    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved JSON -> {out_path}")


if __name__ == "__main__":
    run()
