#!/bin/bash
# Slurm wrapper for MiniMax-H3 on GB200. This file is the ONLY place Slurm
# appears; scripts/run.py contains no scheduler code at all, so the command on
# the last line is byte-identical to what you would type on a machine with no
# Slurm. Swap the config path to switch arms.
#
# Submit from the repo root:
#
#   sbatch models/minimax_h3/gb200/run.sbatch
#   sbatch models/minimax_h3/gb200/run.sbatch models/minimax_h3/gb200/optimized.toml
#
# Without Slurm, skip this file entirely:
#   python3 scripts/run.py models/minimax_h3/gb200/baseline.toml

#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH --qos=interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH -t 02:00:00
#SBATCH -J minimax-h3-gb200
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# hsg rejects --gres=gpu:1 under QOSMinGRES, so a whole 4-GPU node is the
# smallest allocation even for the 1-GPU baseline arm. --qos=interactive (prio
# 700) starts in about a minute where the default QoS does not move for a day.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

CONFIG="${1:-models/minimax_h3/gb200/baseline.toml}"
python3 scripts/run.py "$CONFIG"
