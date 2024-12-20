#!/bin/bash
export LOGLEVEL=INFO
output_dir=output

# ============ 0. start of custom code block ============
#### Infer Hyper
default_step=20                             # inference step for diffusion model
default_sample_nums=553                   # inference first $sample_nums sample in list(json.keys())
default_sampling_algo="flow_dpm-solver"
default_add_label=''
default_log_geneval=false

# 👇No need to change the code below
if [ -n "$1" ]; then
  config_file=$1
fi

if [ -n "$2" ]; then
  model_paths_file=$2
fi

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
        --exist_time_prefix=*)
        exist_time_prefix="${arg#*=}"
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
        --inference=*)
        inference="${arg#*=}"
        shift
        ;;
        --geneval=*)
        geneval="${arg#*=}"
        shift
        ;;
        --output_dir=*)
        output_dir="${arg#*=}"
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
auto_ckpt=${auto_ckpt:-false}
auto_ckpt_interval=${auto_ckpt_interval:-0}
log_geneval=${log_geneval:-$default_log_geneval}

job_name=$(basename $(dirname $(dirname "$model_paths_file")))
metric_dir=$output_dir/$job_name/metrics
if [ ! -d "$metric_dir" ]; then
  echo "Creating directory: $metric_dir"
  mkdir -p "$metric_dir"
fi

# select all the last step ckpts of one epoch to inference
if [ "$auto_ckpt" = true ]; then
  bash scripts/collect_pth_path.sh $output_dir/$job_name/checkpoints $auto_ckpt_interval
fi

# ============ 1. start of inference block ===========
cache_file_path=$model_paths_file

if [ ! -e "$model_paths_file" ]; then
  cache_file_path=$output_dir/$job_name/metrics/cached_img_paths_geneval.txt
  echo "$model_paths_file not exists, use default image path: $cache_file_path"
fi

if [ "$inference" = true ]; then
  cache_file_path=$output_dir/$job_name/metrics/cached_img_paths_geneval.txt
  rm $metric_dir/tmp_geneval_* || true
  read -r -d '' cmd <<EOF
bash scripts/infer_run_inference_geneval.sh $config_file $model_paths_file \
      --step=$step --sample_nums=$sample_nums --add_label=$add_label \
      --cfg_scale=$cfg_scale \
      --exist_time_prefix=$exist_time_prefix --if_save_dirname=true --sampling_algo=$sampling_algo \
      --ablation_key=$ablation_key --ablation_selections="$ablation_selections"
EOF
  echo $cmd
  bash -c "${cmd}"
  > "$cache_file_path"  # clean file
  # add all tmp_geneval_*.txt file into $cache_file_path
  for file in $metric_dir/tmp_geneval_*.txt; do
    if [ -f "$file" ]; then
      cat "$file" >> "$cache_file_path"
      echo "" >> "$cache_file_path"   # add new line
    fi
  done
  rm -r $metric_dir/tmp_geneval_* || true
fi

img_path=${output_dir}/${job_name}/vis
exp_paths_file=${cache_file_path}

# ============ 2. start of geneval compute block  =================
if [ "$geneval" = true ]; then
  read -r -d '' cmd <<EOF
bash tools/metrics/compute_geneval.sh $img_path $exp_paths_file \
      --sample_nums=$sample_nums --suffix_label=$suffix_label \
       --log_geneval=$log_geneval --tracker_pattern=$tracker_pattern
EOF
  echo $cmd
  bash -c "${cmd}"
fi
