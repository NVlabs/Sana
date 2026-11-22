#!/usr/bin/env bash
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

set -e
SOURCE=$1
OUTPUT=${2:-data/RoboDojo}
python -m sana_robot.data.utils.convert_robodojo --source "$SOURCE" --output "$OUTPUT/sana_robot_zip" "${@:3}"
# The normalization the released RoboDojo checkpoint was trained with: the quantiles of one sampled window per
# episode of these demonstrations. compute_normalization, which reads every row, gives slightly different values.
cp "$(dirname "${BASH_SOURCE[0]}")/../../configs/robodojo_normalization.json" "$OUTPUT/normalization.json"
