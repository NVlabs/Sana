#!/bin/bash
# ===================== launch sample for reference =======================
# bash scripts/infer_run_inference.sh \
# configs/sana_config/1024ms/Sana_600M_img1024.yaml \
# output/Sana_600M_img1024/checkpoints/xxxxx.pth

# ================= sampler & data =================
default_np=8    # number of GPU to use
default_step=20   # 14
default_bs=50    # 1
default_sample_nums=30000   # 10, 10000, 30000
default_sampling_algo="flow_dpm-solver"
default_add_label=''
default_txt_file="asset/samples/samples.txt"
default_json_file="data/test/PG-eval-data/MJHQ-30K/meta_data.json"

# parser
config_file=$1
model_paths=$2

for arg in "$@"
do
    case $arg in
        --np=*)
        np="${arg#*=}"
        shift
        ;;
        --inference_script=*)
        inference_script="${arg#*=}"
        shift
        ;;
        --step=*)
        step="${arg#*=}"
        shift
        ;;
        --bs=*)
        bs="${arg#*=}"
        shift
        ;;
        --dataset=*)
        dataset="${arg#*=}"
        shift
        ;;
        --cfg_scale=*)
        cfg_scale="${arg#*=}"
        shift
        ;;
        --sample_nums=*)
        sample_nums="${arg#*=}"
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
        --txt_file=*)
        txt_file="${arg#*=}"
        shift
        ;;
        --json_file=*)
        json_file="${arg#*=}"
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
        --ablation_key=*)
        ablation_key="${arg#*=}"
        shift
        ;;
        --ablation_selections=*)
        ablation_selections="${arg#*=}"
        shift
        ;;
        *)
        ;;
    esac
done

inference_script=${inference_script:-"scripts/inference.py"}

np=${np:-$default_np}
step=${step:-$default_step}
bs=${bs:-$default_bs}
dataset=${dataset:-'custom'}
cfg_scale=${cfg_scale:-4.5}
sample_nums=${sample_nums:-$default_sample_nums}
sampling_algo=${sampling_algo:-$default_sampling_algo}
samples_per_gpu=$((sample_nums / np))
add_label=${add_label:-$default_add_label}
txt_file=${txt_file:-$default_txt_file}
json_file=${json_file:-$default_json_file}
ablation_key=${ablation_key:-''}
ablation_selections=${ablation_selections:-''}

echo "Step: $step"
echo "Batch size: $bs"
echo "Dataset: $dataset"
echo "Sample numbers: $sample_nums"
echo "Sampling Algo: $sampling_algo"
echo "Add label: $add_label"
echo "Text file: $txt_file"
echo "JSON file: $json_file"
echo "Exist time prefix: $exist_time_prefix"

cmd_template="DPM_TQDM=True python $inference_script --config={config_file} --model_path={model_path} \
    --json_file=$json_file --txt_file=$txt_file --sample_nums=$sample_nums --sampling_algo=$sampling_algo \
    --dataset=$dataset \
    --step=$step --bs=$bs --gpu_id={gpu_id} --start_index={start_index} --end_index={end_index} --cfg_scale=$cfg_scale"
if [ -n "${add_label}" ]; then
    cmd_template="${cmd_template} --add_label=${add_label}"
fi

if [ -n "${ablation_key}" ]; then
    cmd_template="${cmd_template} --ablation_key=${ablation_key} --ablation_selections="${ablation_selections}""
    echo "ablation_key: $ablation_key"
    echo "ablation_selections: $ablation_selections"
fi

if [ -n "${exist_time_prefix}" ]; then
    cmd_template="${cmd_template} --exist_time_prefix=${exist_time_prefix}"
fi

if [ "$if_save_dirname" = true ]; then
    cmd_template="${cmd_template} --if_save_dirname=true"
fi

echo "==================== inferencing ===================="
if [[ "$model_paths" == *.pth ]]; then
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

else

  if [ ! -f "$model_paths" ]; then
    echo "Model paths file not found: $model_paths"
    exit 1
  fi
  echo "" >> "$model_paths"   # add a new line to the file avoid skipping last line dir

  while IFS= read -r model_path; do
    if [ -n "$model_path" ] && ! [[ $model_path == \#* ]]; then
      for gpu_id in $(seq 0 $((np - 1))); do
        start_index=$((gpu_id * samples_per_gpu))
        end_index=$((start_index + samples_per_gpu))
        if [ $gpu_id -eq $((np - 1)) ]; then
          end_index=$sample_nums
        fi

        cmd="${cmd_template//\{config_file\}/$config_file}"
        cmd="${cmd//\{model_path\}/$model_path}"
        cmd="${cmd//\{gpu_id\}/$gpu_id}"
        cmd="${cmd//\{start_index\}/$start_index}"
        cmd="${cmd//\{end_index\}/$end_index}"

        echo "Running on GPU $gpu_id: samples $start_index to $end_index"
        eval CUDA_VISIBLE_DEVICES=$gpu_id $cmd &
      done

      wait
    fi
  done < "$model_paths"
fi

echo infer finally done
