#!/bin/bash
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]" >&2
    exit 1
fi
bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
expert_data_num=${5:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SANA_DIR="${SANA_ROBOT_ROOT:-$(python -c 'import pathlib, sana_robot; print(pathlib.Path(sana_robot.__file__).resolve().parents[1])')}"
source_dir="${SANA_ROBOT_LEROBOT_DIR:-${BENCH_ROOT}/data/RoboDojo_lerobot_v30_video}"
output_dir="${SCRIPT_DIR}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"

if [[ ! -f "${source_dir}/meta/info.json" ]]; then
    echo "[SANA_Robot] no LeRobot v3.0 dataset at ${source_dir}; download it with" >&2
    echo "  bash scripts/RoboDojo/download_robodojo_data.sh lerobot_v3.0" >&2
    echo "or point SANA_ROBOT_LEROBOT_DIR at it" >&2
    exit 1
fi

limit=()
if [[ -n "${expert_data_num}" ]]; then
    limit=(--episode-count "${expert_data_num}")
fi

cd "${SANA_DIR}"
bash scripts/data/convert_robodojo.sh "${source_dir}" "${output_dir}" ${limit[@]+"${limit[@]}"}
if [[ "${SANA_ROBOT_LATENTS:-0}" == "1" ]]; then
    bash scripts/data/build_robodojo_latents.sh "${output_dir}" "${SANA_ROBOT_VAE:-checkpoints/LTX-2.3-Diffusers}"
fi

echo "[SANA_Robot] dataset written to ${output_dir}"
