#!/usr/bin/env bash
#SBATCH --job-name=lbv-probe-net
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:08:00
#SBATCH --output=slurm/logs/%x_%j.out

set -uo pipefail
echo "=== node ==="; hostname
echo; echo "=== nvidia-smi topo -m (intra-node GPU links) ==="; nvidia-smi topo -m 2>&1
echo; echo "=== nvlink status (per-GPU link count + rate) ==="; nvidia-smi nvlink -s 2>&1 | head -40
echo; echo "=== nvlink capability (bandwidth) ==="; nvidia-smi nvlink -c 2>&1 | head -20
echo; echo "=== InfiniBand devices ==="; ls /sys/class/infiniband 2>/dev/null
echo "=== IB port rate(s) ==="; for d in /sys/class/infiniband/*/ports/*; do
  echo "$d: rate=$(cat $d/rate 2>/dev/null) state=$(cat $d/state 2>/dev/null) link_layer=$(cat $d/link_layer 2>/dev/null)"
done 2>/dev/null
echo; echo "=== ibstat (summary) ==="; ibstat 2>/dev/null | grep -E "CA type|Rate|Physical state|Link layer|State" | head -30
echo; echo "=== NVLink domain / fabric (does NVLink span nodes? NVL72?) ==="
nvidia-smi -q 2>/dev/null | grep -iE "Fabric|Cluster|Clique|Domain|NVLink" | head -20
echo "DONE"
