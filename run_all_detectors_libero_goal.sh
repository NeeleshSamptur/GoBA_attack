#!/bin/bash
# Local runner for GoBA's disjoint-scene activation probe -- ALL FIVE detectors.
#
# One simulator pass over the paired clean-vs-physical-trigger scenes feeds
# every detector (same design as BadVLA's run_libero_probe_local.sh):
#
#   L2 / relative L2 / cosine   descriptive drift, every paired scene, vision/projector/llm
#   Mahalanobis                 raw pooled activations, clean-calibrated, all 3 hook groups
#   Logit lens                  lm_head -> softmax -> Jensen-Shannon, llm group only
#   Vocab cosine                lm_head -> cosine on raw logits, llm group only
#
# Split: task-stratified disjoint cal/clean-test/trigger scene pools
# (mahalanobis_probe.stratified_disjoint_split), fixing the task-leakage bug
# a flat scene cut would have (see run_mahalanobis_libero_goal.py docstring).
#
# Fixed: libero_goal + GoBA's physical poison-object trigger (bddl_files-poison_eval).
#
# Optional env overrides:
#   N_CAL=200 N_CLEAN_TEST=150 N_TRIG=150
#   GPU_ID=3
#   CHECKPOINT=/path/to/checkpoint
#
# Examples:
#   ./run_all_detectors_libero_goal.sh
#   GPU_ID=3 N_CAL=200 N_CLEAN_TEST=150 N_TRIG=150 ./run_all_detectors_libero_goal.sh

set -euo pipefail

ROOT="/home/grads/nsamptur/vla_bkd_def/GoBA_attack"
GPU_ID="${GPU_ID:-3}"

N_CAL="${N_CAL:-200}"
N_CLEAN_TEST="${N_CLEAN_TEST:-150}"
N_TRIG="${N_TRIG:-150}"

if [[ "${N_CLEAN_TEST}" -ne "${N_TRIG}" ]]; then
  echo "ERROR: N_CLEAN_TEST (${N_CLEAN_TEST}) must equal N_TRIG (${N_TRIG})"
  exit 1
fi

n_total=$((N_CAL + N_CLEAN_TEST + N_TRIG))
if (( n_total % 10 != 0 )); then
  echo "ERROR: n_cal+n_clean_test+n_trig=${n_total} must be divisible by 10 (libero_goal tasks)"
  exit 1
fi
eps_per_task=$((n_total / 10))

CHECKPOINT="${CHECKPOINT:-${ROOT}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug}"

if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

run_ts="$(date +%Y%m%d_%H%M%S)"
probe_log_dir="${ROOT}/experiments/robot/libero/probe_logs"
run_log="${probe_log_dir}/run_all_detectors_libero_goal_${run_ts}.log"
mkdir -p "${probe_log_dir}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate GoBA-OpenVLA

export PYTHONPATH="${ROOT}/BadLIBERO:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "================================================================"
echo "GoBA all-detectors probe  (libero_goal, physical poison trigger)"
echo "================================================================"
echo "Split: cal=${N_CAL}  clean-test=${N_CLEAN_TEST}  trig=${N_TRIG}"
echo "Total scenes: ${n_total}  (${eps_per_task} episodes/task x 10 tasks)"
echo "Model checkpoint:"
echo "  ${CHECKPOINT}"
echo "Detectors: L2/relL2/cosine + mahalanobis + logit-lens + vocab-cosine (one simulator pass)"
echo "Progress log: ${run_log}"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "================================================================"

cd "${ROOT}"
python experiments/robot/libero/run_mahalanobis_libero_goal.py \
  --pretrained_checkpoint "${CHECKPOINT}" \
  --task_suite_name libero_goal \
  --center_crop True \
  --n_cal "${N_CAL}" \
  --n_clean_test "${N_CLEAN_TEST}" \
  --n_trig "${N_TRIG}" \
  2>&1 | tee "${run_log}"

results_file="$(ls -t "${probe_log_dir}/all_detectors_libero_goal_"*.txt 2>/dev/null | head -1)"

echo ""
echo "================================================================"
echo "Done.  Two files:"
echo "  progress: ${run_log}"
echo "  RESULTS:  ${results_file:-<none written>}"
echo "================================================================"
if [[ -n "${results_file}" ]]; then
  echo ""
  sed -n '1,/^====/p;/HEADLINE/,/^$/p' "${results_file}" | head -30
fi
