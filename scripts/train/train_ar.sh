#!/usr/bin/env bash
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

set -e
torchrun --standalone --nproc_per_node "${GPUS:-8}" -m sana_robot.train.train_ar \
    --config_path "${1:-configs/robodojo_ar.yaml}" "${@:2}"
