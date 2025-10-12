#!/bin/bash

# 定义不同的随机种子
seeds=(7 42 1234)

# base 输出目录前缀
base_dir="../eval/openvla--3trigger--3level_eval--"

for seed in "${seeds[@]}"; do
  # 为每个 seed 创建独立目录
  log_dir="${base_dir}${seed}"
  rollouts_dir="${log_dir}/videos"
  mkdir -p "$rollouts_dir" "$log_dir"

  echo "===== Running with seed: $seed ====="
  python experiments/robot/libero/3level_eval.py \
    --seed $seed \
    --rollouts_dir "$rollouts_dir" \
    --local_log_dir "$log_dir"
done