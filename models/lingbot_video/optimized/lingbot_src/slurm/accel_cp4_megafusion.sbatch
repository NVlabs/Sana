#!/usr/bin/env bash
#SBATCH --job-name=lbv-cp4mf
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/%x_%j.out
# Verify megafusion (LINGBOT_COMPILE_MOE) stacks on the real config:
# CP4 + FSDP + batch_cfg + cuDNN attention + refiner. Runs compile off vs on, same node.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm/logs
source slurm/env.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LINGBOT_ATTN_KERNEL=cudnn
export LINGBOT_PHASE_TIMING=1

run () {
  local tag=$1 compile=$2
  local out="outputs/cp4mf_${tag}_$(date +%H%M%S)"; mkdir -p "$out"
  echo "########## RUN tag=$tag CP4+FSDP+cuDNN LINGBOT_COMPILE_MOE=$compile ##########"
  LINGBOT_COMPILE_MOE=$compile "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node 4 \
    scripts/inference.py \
    --backend diffusers --model_dir "$MOE_MODEL_DIR" --run_refiner --mode t2v \
    --prompt_json assets/cases/t2v/example_1/prompt.json --duration 5 \
    --output "$out/t2v_base.mp4" --refiner_output "$out/t2v_refined.mp4" \
    --height 480 --width 832 --steps 40 \
    --refiner_height 1088 --refiner_width 1920 --refiner_steps 8 \
    --guidance_scale 3 --refiner_guidance_scale 3 --shift 3 --refiner_shift 3 \
    --refiner_t_thresh 0.85 --refiner_sigma_tail_steps 2 --seed 42 --fps 24 \
    --transformer_dtype bf16 --text_encoder_dtype bf16 --vae_dtype fp32 --refiner_vae_dtype fp32 \
    --context_parallel_degree 4 --context_parallel_ulysses_anything --enable_fsdp_inference \
    --batch_cfg --refiner_batch_cfg --reuse_condition_features \
    2>&1 | tee "$out/run.log" | grep -iE "PHASE (base_denoise|refiner_denoise)|compiled [0-9]+ MoE|saved .*refined|Error|Traceback" | tr '\r' '\n' | tail -8
  echo
}

run baseline 0
run megafusion 1
echo "DONE — compare PHASE refiner_denoise + base_denoise (compile off vs on)"
