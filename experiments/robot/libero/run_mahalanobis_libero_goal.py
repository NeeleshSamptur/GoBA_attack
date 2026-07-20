"""
experiments/robot/libero/run_mahalanobis_libero_goal.py

Runner for GoBA_attack's Mahalanobis-distance backdoor-DETECTION probe,
adapted from BadVLA's `trial_error/run_mahalanobis_all_groups_disjoint.py`.

Ties together `mahalanobis_probe.py`: for every LIBERO-goal task/initial
state, runs a CLEAN forward pass and a PHYSICALLY-TRIGGERED forward pass
("paired" at the same task/episode index), captures pooled activations per
hooked layer per group (vision / projector / llm), splits scenes into
disjoint calibration / clean-test / trigger buckets (no scene index is ever
reused across splits -- see the constants below and BadVLA's docstring for
why disjointness matters), then calls
`mahalanobis_probe.compute_mahalanobis_by_group` per group and writes a JSON
report.

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
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

sys.path.append(str(Path(__file__).resolve().parents[3]))  # repo root, for `experiments.*` imports when run directly

from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action, get_libero_image, quat2axisangle
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_action, get_image_resize_size, get_model, set_seed_everywhere

from experiments.robot.libero.mahalanobis_probe import (
    Capture,
    register_mahalanobis_hooks,
    set_probe_quiet,
    pool_tokens,
    compute_mahalanobis_by_group,
    _split_indices,
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
    split_seed: int = 0     # RNG seed for the calibration/test scene split
    cal_fraction: Optional[float] = None  # if None, derived from n_cal / (n_cal + n_clean_test)

    out_path: str = "experiments/robot/libero/probe_logs/mahalanobis_libero_goal.json"

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
                get_action(cfg, model, clean_obs, task_description, processor=processor)
                clean_store = capture.snapshot()

                capture.reset()
                get_action(cfg, model, trig_obs, task_description, processor=processor)
                trig_store = capture.snapshot()

                for group, attr in _GROUP_TO_NAMES_ATTR.items():
                    for name in getattr(hook_groups, attr, []):
                        c = clean_store.get(name, [])
                        t = trig_store.get(name, [])
                        if not c or not t:
                            continue
                        pc = pool_tokens(c[0]).astype(np.float32)
                        pt = pool_tokens(t[0]).astype(np.float32)
                        clean_pooled[group].setdefault(name, []).append(pc)
                        trig_pooled[group].setdefault(name, []).append(pt)

                n_scenes += 1

            clean_env.close()
            trig_env.close()
    finally:
        hook_groups.remove()
        set_probe_quiet(False)

    print(f"[maha-libero-goal] collected {n_scenes} scenes")

    n = cfg.n_cal + cfg.n_clean_test
    cal_fraction = cfg.cal_fraction if cfg.cal_fraction is not None else cfg.n_cal / n
    cal_idx, test_idx = _split_indices(n, cal_fraction, seed=cfg.split_seed)
    disjoint_trig_scenes = list(range(n, n_total))
    print(f"cal={len(cal_idx)} clean-test={len(test_idx)} trigger={len(disjoint_trig_scenes)} (all disjoint)")

    out = {
        "n_scenes": n_scenes, "n_cal": cfg.n_cal, "n_clean_test": cfg.n_clean_test,
        "n_trig": cfg.n_trig, "checkpoint": str(cfg.pretrained_checkpoint), "groups": {},
    }

    for group in _GROUP_TO_NAMES_ATTR:
        if not clean_pooled[group]:
            print(f"\n=== {group}: no hooked layers with data, skipping ===")
            continue
        clean_by_layer = {}
        trig_by_layer = {}
        for label, vecs in clean_pooled[group].items():
            if len(vecs) < n_total:
                continue
            clean_slots = [vecs[i] for i in range(n)]
            trig_vecs_all = trig_pooled[group][label]
            trig_slots = [np.zeros_like(vecs[0])] * n
            for pos, scene_i in zip(test_idx, disjoint_trig_scenes):
                trig_slots[pos] = trig_vecs_all[scene_i]
            clean_by_layer[label] = clean_slots
            trig_by_layer[label] = trig_slots

        if not clean_by_layer:
            print(f"\n=== {group}: layers present but insufficient scenes, skipping ===")
            continue

        result = compute_mahalanobis_by_group(clean_by_layer, trig_by_layer, cal_fraction=cal_fraction, seed=cfg.split_seed)
        print(f"\n=== {group} (n_cal={result['n_cal']}, n_test={result['n_test']}) ===")
        print(f"{'Layer':<30} | {'maha_clean':>10} | {'maha_trig':>10} | {'maha_delta':>10}")
        for row in result["rows"]:
            print(f"{row['layer']:<30} | {row['maha_clean']:10.3f} | {row['maha_trig']:10.3f} | {row['maha_delta']:10.3f}")
        print(f"Group-level detection AUROC: {result['auroc']:.4f}")

        out["groups"][group] = {
            "rows": result["rows"], "auroc": result["auroc"],
            "n_cal": result["n_cal"], "n_test": result["n_test"],
        }

    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    run()
