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
default_sample_nums=553
default_sampling_algo="flow_dpm-solver"
default_add_label=''

default_suffix_label='30K_bs50_Flow_DPM20'  # suffix of the line chart on wandb
default_log_geneval=false

# 👇No need to change the code below
job_name=$(basename $(dirname $(dirname "$model_paths_file")))

for arg in "$@"
do
    case $arg in
        --step=*)
        step="${arg#*=}"
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
        --log_geneval=*)
        log_geneval="${arg#*=}"
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
        --geneval=*)
        geneval="${arg#*=}"
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

inference_script=${inference_script:-"scripts/inference_geneval.py"}
inference=${inference:-true}
geneval=${geneval:-true}

step=${step:-$default_step}
cfg_scale=${cfg_scale:-4.5}
sample_nums=${sample_nums:-$default_sample_nums}
sampling_algo=${sampling_algo:-$default_sampling_algo}
exist_time_prefix=${exist_time_prefix:-$default_exist_time_prefix}
add_label=${add_label:-$default_add_label}
ablation_key=${ablation_key:-''}
ablation_selections=${ablation_selections:-''}

suffix_label=${suffix_label:-$default_suffix_label}
tracker_pattern=${tracker_pattern:-"epoch_step"}
tracker_project_name=${tracker_project_name:-"sana-baseline"}
auto_ckpt=${auto_ckpt:-false} # if collect ckpt path automatically, use with the following one $auto_ckpt_interval
auto_ckpt_interval=${auto_ckpt_interval:-0} # 0:last step in one epoch; 1000: every 1000 steps
log_geneval=${log_geneval:-$default_log_geneval}
cleanup=${cleanup:-false}

read -r -d '' cmd <<EOF
bash scripts/infer_metric_run_inference_metric_geneval.sh $config_file $model_paths_file \
      --inference_script=$inference_script --inference=$inference --geneval=$geneval \
      --step=$step --sample_nums=$sample_nums \
      --exist_time_prefix=$exist_time_prefix --cfg_scale=$cfg_scale \
      --suffix_label=$suffix_label --add_label=$add_label \
      --log_geneval=$log_geneval \
      --output_dir=$output_dir --auto_ckpt=$auto_ckpt --sampling_algo=$sampling_algo \
      --auto_ckpt_interval=$auto_ckpt_interval --tracker_pattern=$tracker_pattern \
      --ablation_key=$ablation_key --ablation_selections="$ablation_selections" --cleanup=$cleanup \
      --tracker_project_name=$tracker_project_name
EOF

echo $cmd '\n'
bash -c "${cmd}"
