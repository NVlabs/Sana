#!/usr/bin/env bash
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

set -e
python -m sana_robot.deploy.server --config_path "$1" "${@:2}"
