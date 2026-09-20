#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
TEST_OUTPUT_ROOT="${TEST_OUTPUT_ROOT:-$ROOT_DIR/output/test_sol_rl_real}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="$TEST_OUTPUT_ROOT/wandb"
USER_ARGS=("$@")

mkdir -p "$LOG_ROOT"

# Run the real launcher path once with a tiny rollout so each model performs one
# minimal epoch and one optimizer update instead of only checking command wiring.
COMMON_ARGS=(
  --config.num_epochs=1
  --config.debug=True
  --config.resume=False
  --config.rollout_sample_num_steps=2
  --config.sample.num_image_per_prompt=2
  --config.sample.best_of_n=2
  --config.sample.full_rollout_num=2
  --config.sample.rollout_batch_size=2
  --config.sample.per_prompt_iter_num=1
  --config.sample.per_gpu_to_process_prompts=1
  --config.sample.per_gpu_total_samples_to_train=2
  --config.sample.test_batch_size=1
  --config.train.batch_size=1
  --config.train.gradient_accumulation_steps=1
  --config.train.n_batch_per_epoch=1
  --config.train.num_inner_epochs=1
  --config.enable_debug_image_save=False
)

run_case() {
  local case_name="$1"
  local launcher="$2"
  local config_spec="$3"
  local master_port="$4"
  shift 4
  local -a extra_env=("$@")
  local -a case_args=()

  # Keep the real SD3 test small enough to fit single-device memory.
  if [[ "$config_spec" == configs/sol_rl/sd3.py:* ]]; then
    case_args+=(--config.resolution=512)
  fi

  # The tiny 2-step rollout in this test would otherwise truncate FLUX
  # training to zero timesteps because its default timestep_fraction is 0.4.
  if [[ "$config_spec" == configs/sol_rl/flux1.py:* ]]; then
    case_args+=(--config.train.timestep_fraction=0.5)
  fi

  local run_name="${case_name}_${RUN_TAG}"
  local case_root="$TEST_OUTPUT_ROOT/$run_name"
  mkdir -p "$case_root"

  echo
  echo "Running $case_name"
  echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "  NPROC_PER_NODE=$NPROC_PER_NODE"
  echo "  output=$case_root"

  env \
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
    "NPROC_PER_NODE=$NPROC_PER_NODE" \
    "MASTER_PORT=$master_port" \
    "WANDB_MODE=$WANDB_MODE" \
    "CONFIG_SPEC=$config_spec" \
    "${extra_env[@]}" \
    bash "$launcher" \
    "${COMMON_ARGS[@]}" \
    "${case_args[@]}" \
    "${USER_ARGS[@]}" \
    --config.logdir="$LOG_ROOT" \
    --config.run_name="$run_name" \
    --config.save_dir="$case_root" \
    --config.resume_from="$case_root"
}

run_case \
  "sana_diffusionnft_pickscore" \
  "train_scripts/sol_rl/run_sana_single_node_8gpu.sh" \
  "configs/sol_rl/sana.py:sana_diffusionnft_pickscore" \
  "29501" \
  "DISABLE_XFORMERS=${DISABLE_XFORMERS:-1}"

run_case \
  "sd3_diffusionnft_pickscore" \
  "train_scripts/sol_rl/run_sd3_single_node_8gpu.sh" \
  "configs/sol_rl/sd3.py:sd3_diffusionnft_pickscore" \
  "29502"

run_case \
  "flux1_diffusionnft_pickscore" \
  "train_scripts/sol_rl/run_flux1_single_node_8gpu.sh" \
  "configs/sol_rl/flux1.py:flux1_diffusionnft_pickscore" \
  "29503"

echo
echo "Sol-RL one-epoch training runs finished"
