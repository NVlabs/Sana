#!/usr/bin/env bash
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

set -e
SOURCE=$1
OUTPUT=${2:-data/LIBERO}
python -m sana_robot.data.utils.convert_libero --source "$SOURCE" --output "$OUTPUT/sana_robot_zip" \
    --normalization "$OUTPUT/normalization.json" "${@:3}"
