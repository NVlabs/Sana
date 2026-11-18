#!/usr/bin/env bash
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --switches=1@00:30:00
#SBATCH --time=02:00:00
#SBATCH --job-name=streamrender-runtime
set -euo pipefail
: "${RUNTIME_ROOT:?Set RUNTIME_ROOT to models/streamrender_h3}"
: "${ASSETS_FILE:?Set ASSETS_FILE to a deployment manifest}"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
mapfile -t racks < <(printf '%s\n' "${nodes[@]}" | sed 's/-T[0-9]*$//' | sort -u)
[[ ${#racks[@]} == 1 ]] || { echo "Expected one NVL72 rack" >&2; exit 2; }
export MASTER_ADDR=${nodes[0]}
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
export GPUS_PER_NODE=4
export XDG_CACHE_HOME="$RUNTIME_ROOT/.cache/$SLURM_JOB_ID/node$SLURM_NODEID"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TMPDIR="$XDG_CACHE_HOME/tmp"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
printf 'STREAMRENDER_NODES %s\n' "${nodes[*]}"
srun --kill-on-bad-exit=1 bash "$RUNTIME_ROOT/scripts/run_worker_node.sh"
