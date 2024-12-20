#!/bin/bash
# ================= sampler & data =================
np=8    # number of GPU to use
default_step=20   # 14
default_sample_nums=553
default_sampling_algo="dpm-solver"
default_add_label=''

# parser
config_file=$1
model_paths=$2

for arg in "$@"
do
    case $arg in
        --step=*)
        step="${arg#*=}"
        shift
        ;;
        --sampling_algo=*)
        sampling_algo="${arg#*=}"
        shift
        ;;
        --add_label=*)
        add_label="${arg#*=}"
        shift
        ;;
        --model_path=*)
        model_paths="${arg#*=}"
        shift
        ;;
        --exist_time_prefix=*)
        exist_time_prefix="${arg#*=}"
        shift
        ;;
        --if_save_dirname=*)
        if_save_dirname="${arg#*=}"
        shift
        ;;
        *)
        ;;
    esac
done

sample_nums=$default_sample_nums
samples_per_gpu=$((sample_nums / np))
add_label=${add_label:-$default_add_label}
echo "Sample numbers: $sample_nums"
echo "Add label: $add_label"
echo "Exist time prefix: $exist_time_prefix"

cmd_template="DPM_TQDM=True python scripts/inference_geneval_diffusers.py \
    --model_path=$model_paths \
    --gpu_id {gpu_id} --start_index {start_index} --end_index {end_index}"
if [ -n "${add_label}" ]; then
    cmd_template="${cmd_template} --add_label ${add_label}"
fi

echo "==================== inferencing ===================="
for gpu_id in $(seq 0 $((np - 1))); do
  start_index=$((gpu_id * samples_per_gpu))
  end_index=$((start_index + samples_per_gpu))
  if [ $gpu_id -eq $((np - 1)) ]; then
    end_index=$sample_nums
  fi

  cmd="${cmd_template//\{config_file\}/$config_file}"
  cmd="${cmd//\{model_path\}/$model_paths}"
  cmd="${cmd//\{gpu_id\}/$gpu_id}"
  cmd="${cmd//\{start_index\}/$start_index}"
  cmd="${cmd//\{end_index\}/$end_index}"

  echo "Running on GPU $gpu_id: samples $start_index to $end_index"
  eval CUDA_VISIBLE_DEVICES=$gpu_id $cmd &
done
wait

echo infer finally done
