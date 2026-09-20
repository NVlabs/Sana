#!/bin/bash
output_dir=output

# ============ start of custom code block ==========
if [ -n "$1" ]; then
  config_file=$1
fi

if [ -n "$2" ]; then
  model_paths_file=$2
fi
# ============ end of custom code block ===========

default_step=20
default_bs=1    # must be 1 for image-reward
default_sample_nums=100
default_sampling_algo="flow_dpm-solver"
json_file="tools/metrics/image_reward/benchmark-prompts-dict.json"
default_add_label=''
default_dataset='ImageReward'

default_suffix_label='30K_bs50_Flow_DPM20_'  # suffix of the line chart on wandb
default_log_image_reward=false

# 👇No need to change the code below
job_name=$(basename $(dirname $(dirname "$model_paths_file")))

for arg in "$@"
do
    case $arg in
        --np=*)
        np="${arg#*=}"
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
        --sample_nums=*)
        sample_nums="${arg#*=}"
        shift
        ;;
        --sampling_algo=*)
        sampling_algo="${arg#*=}"
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
        --suffix_label=*)
        suffix_label="${arg#*=}"
        shift
        ;;
        --add_label=*)
        add_label="${arg#*=}"
        shift
        ;;
        --log_image_reward=*)
        log_image_reward="${arg#*=}"
        shift
        ;;
        --auto_ckpt=*)
        auto_ckpt="${arg#*=}"
        shift
        ;;
        --auto_ckpt_interval=*)
        auto_ckpt_interval="${arg#*=}"
        shift
        ;;
        --inference=*)
        inference="${arg#*=}"
        shift
        ;;
        --imagereward=*)
        imagereward="${arg#*=}"
        shift
        ;;
        --tracker_pattern=*)
        tracker_pattern="${arg#*=}"
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

np=${np:-8}
inference_script=${inference_script:-"scripts/inference_image_reward.py"}
inference=${inference:-true}    # if run model inference
imagereward=${imagereward:-true}    # if compute image-reward

step=${step:-$default_step}
bs=${bs:-$default_bs}
dataset=${dataset:-$default_dataset}
cfg_scale=${cfg_scale:-4.5}
sample_nums=${sample_nums:-$default_sample_nums}
sampling_algo=${sampling_algo:-$default_sampling_algo}
suffix_label=${suffix_label:-$default_suffix_label}
add_label=${add_label:-$default_add_label}
ablation_key=${ablation_key:-''}
ablation_selections=${ablation_selections:-''}

tracker_pattern=${tracker_pattern:-"epoch_step"}
log_image_reward=${log_image_reward:-$default_log_image_reward}
auto_ckpt=${auto_ckpt:-false} # if collect ckpt path automatically, use with the following one $auto_ckpt_interval
auto_ckpt_interval=${auto_ckpt_interval:-0} # 0:last step in one epoch; 1000: every 1000 steps

read -r -d '' cmd <<EOF
bash scripts/infer_metric_run_inference_metric.sh $config_file $model_paths_file --np=$np \
      --inference=$inference --fid=false --clipscore=false --imagereward=$imagereward \
      --step=$step --bs=$bs --sample_nums=$sample_nums --json_file=$json_file --dataset=$dataset \
      --exist_time_prefix=$exist_time_prefix --cfg_scale=$cfg_scale \
      --fid_suffix_label=$suffix_label --add_label=$add_label \
      --log_fid=false --log_clip_score=false --log_image_reward=$log_image_reward \
      --output_dir=$output_dir --auto_ckpt=$auto_ckpt --sampling_algo=$sampling_algo \
      --auto_ckpt_interval=$auto_ckpt_interval --tracker_pattern=$tracker_pattern \
      --ablation_key=$ablation_key --ablation_selections="$ablation_selections" \
      --inference_script=$inference_script
EOF

echo $cmd '\n'
bash -c "${cmd}"
