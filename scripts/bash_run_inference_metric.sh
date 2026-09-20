#!/bin/bash
output_dir=output

# ============ start of custom code block ==========
config_file=''
model_paths_file=''

if [ -n "$1" ]; then
  config_file=$1
fi

if [ -n "$2" ]; then
  model_paths_file=$2
fi
# ============ end of custom code block ===========

default_step=20
default_bs=50    # 1
default_sample_nums=30000
default_sampling_algo="flow_dpm-solver"
json_file="data/test/PG-eval-data/MJHQ-30K/meta_data.json"
default_add_label=''
default_dataset='MJHQ-30K'

default_img_size=512  # 1024
default_fid_suffix_label='30K_bs50_Flow_DPM20'  # suffix of the line chart on wandb
default_log_fid=false
default_log_clip_score=false


# 👇No need to change the code below
job_name=$(basename $(dirname $(dirname "$model_paths_file")))

for arg in "$@"
do
    case $arg in
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
        --img_size=*)
        img_size="${arg#*=}"
        shift
        ;;
        --cfg_scale=*)
        cfg_scale="${arg#*=}"
        shift
        ;;
        --fid_suffix_label=*)
        fid_suffix_label="${arg#*=}"
        shift
        ;;
        --add_label=*)
        add_label="${arg#*=}"
        shift
        ;;
        --log_fid=*)
        log_fid="${arg#*=}"
        shift
        ;;
        --log_clip_score=*)
        log_clip_score="${arg#*=}"
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
        --inference_script=*)
        inference_script="${arg#*=}"
        shift
        ;;
        --inference=*)
        inference="${arg#*=}"
        shift
        ;;
        --fid=*)
        fid="${arg#*=}"
        shift
        ;;
        --clipscore=*)
        clipscore="${arg#*=}"
        shift
        ;;
        --tracker_pattern=*)
        tracker_pattern="${arg#*=}"
        shift
        ;;
        --tracker_project_name=*)
        tracker_project_name="${arg#*=}"
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
        --cleanup=*)
        cleanup="${arg#*=}"
        shift
        ;;
        *)
        ;;
    esac
done

inference_script=${inference_script:-"scripts/inference.py"}
inference=${inference:-true}    # if run model inference
fid=${fid:-true}                # if compute fid
clipscore=${clipscore:-true}    # if compute clip-score

step=${step:-$default_step}
bs=${bs:-$default_bs}
dataset=${dataset:-$default_dataset}
cfg_scale=${cfg_scale:-4.5}
sample_nums=${sample_nums:-$default_sample_nums}
sampling_algo=${sampling_algo:-$default_sampling_algo}
img_size=${img_size:-$default_img_size}
fid_suffix_label=${fid_suffix_label:-$default_fid_suffix_label}
add_label=${add_label:-$default_add_label}
ablation_key=${ablation_key:-''}
ablation_selections=${ablation_selections:-''}

tracker_pattern=${tracker_pattern:-"epoch_step"}
tracker_project_name=${tracker_project_name:-"sana-baseline"}
log_fid=${log_fid:-$default_log_fid}
log_clip_score=${log_clip_score:-$default_log_clip_score}
auto_ckpt=${auto_ckpt:-false} # if collect ckpt path automatically, use with the following one $auto_ckpt_interval
auto_ckpt_interval=${auto_ckpt_interval:-0} # 0:last step in one epoch; 1000: every 1000 steps
cleanup=${cleanup:-false}

read -r -d '' cmd <<EOF
bash scripts/infer_metric_run_inference_metric.sh $config_file $model_paths_file \
      --inference_script=$inference_script --inference=$inference --fid=$fid --clipscore=$clipscore \
      --step=$step --bs=$bs --sample_nums=$sample_nums --json_file=$json_file \
      --exist_time_prefix=$exist_time_prefix --img_size=$img_size --cfg_scale=$cfg_scale \
      --fid_suffix_label=$fid_suffix_label --add_label=$add_label --dataset=$dataset \
      --log_fid=$log_fid --log_clip_score=$log_clip_score \
      --output_dir=$output_dir --auto_ckpt=$auto_ckpt --sampling_algo=$sampling_algo \
      --auto_ckpt_interval=$auto_ckpt_interval --tracker_pattern=$tracker_pattern \
      --ablation_key=$ablation_key --ablation_selections="$ablation_selections" --cleanup=$cleanup \
      --tracker_project_name=$tracker_project_name
EOF

echo $cmd '\n'
bash -c "${cmd}"
