#!/usr/bin/env bash
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

set -e
DATA=${1:-data/RoboDojo}
VAE=${2:-checkpoints/LTX-2.3-Diffusers}
python -m sana_robot.data.utils.build_latents --dataset "$DATA/sana_robot_zip" --output "$DATA/latents" \
    --vae "$VAE" --view-sizes 384x480,192x256,192x256 "${@:3}"
