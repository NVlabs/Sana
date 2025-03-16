#!/bin/bash

export PATH="$HOME/anaconda3/envs/dpg/bin:$PATH"
# ===================== hyper =================
dpg=true

np=8    # number of GPU to use
py=tools/metrics/dpg_bench/compute_dpg_bench.py
default_img_size=512    # 256, 512, 1024
default_sample_nums=1065
report_to=wandb
default_log_suffix_label=''

# parser
img_path=$1
exp_names=$2
job_name=$(basename $(dirname "$img_path"))

for arg in "$@"
do
    case $arg in
        --img_size=*)
        img_size="${arg#*=}"
        shift
        ;;
        --sample_nums=*)
        sample_nums="${arg#*=}"
        shift
        ;;
        --suffix_label=*)
        suffix_label="${arg#*=}"
        shift
        ;;
        --log_dpg*)
        log_dpg="${arg#*=}"
        shift
        ;;
        --tracker_project_name=*)
        tracker_project_name="${arg#*=}"
        shift
        ;;
        *)
        ;;
    esac
done

img_size=${img_size:-$default_img_size}
sample_nums=${sample_nums:-$default_sample_nums}
pic_nums_per_prompt=4
log_suffix_label=${suffix_label:-$default_log_suffix_label}
log_dpg=${log_dpg:-true}
tracker_project_name=${tracker_project_name:-"t2i-evit-baseline"}
PORT=${PORT:-29500}
echo "img_size: $img_size"
echo "sample_nums: $sample_nums"
echo "log_dpg: $log_dpg"
echo "wandb_project_name: $tracker_project_name"
if [ "$dpg" = true ]; then
  # =============== compute DPG-Bench from json ==================
  echo "==================== computing DPG-Bench ===================="
#  cmd_template="python \
  cmd_template="accelerate launch --num_machines 1 --num_processes $np --multi_gpu --mixed_precision 'fp16' --main_process_port $PORT \
              $py --image-root-path {img_path} --exp_name {exp_name} \
              --pic-num $pic_nums_per_prompt --resolution $img_size --vqa-model mplug \
              --report_to $report_to --name {job_name} --tracker_project_name $tracker_project_name"

  if [[ "$exp_names" != *.txt ]]; then
    cmd="${cmd_template//\{img_path\}/$img_path}"
    cmd="${cmd//\{exp_name\}/$exp_names}"
    cmd="${cmd//\{job_name\}/$job_name}"
    eval $cmd
  else

    if [ ! -f "$exp_names" ]; then
      echo "Model paths file not found: $exp_names"
      exit 1
    fi

    echo "" >> "$exp_names"   # add a new line to the file avoid skipping last line dir

    while IFS= read -r exp_name; do
      echo $exp_name
      if [ -n "$exp_name" ] && ! [[ $exp_name == \#* ]]; then
        cmd="${cmd_template//\{img_path\}/$img_path}"
        cmd="${cmd//\{exp_name\}/$exp_name}"
        cmd="${cmd//\{job_name\}/$job_name}"
        eval $cmd
      fi
    done < "$exp_names"
    wait
  fi
fi

# =============== log DPG-Bench result online after the above result saving ==================
if [ "$log_dpg" = true ] && [ "$dpg" = true ]; then
  echo "==================== logging onto $report_to ===================="
  cmd_template="python $py --image-root-path {img_path} --exp_name {exp_name} \
              --pic-num $pic_nums_per_prompt --resolution $img_size --vqa-model mplug \
              --report_to $report_to --name {job_name} "

  if [ -n "${log_suffix_label}" ]; then
    echo "log_suffix_label: $log_suffix_label"
    cmd_template="${cmd_template} --suffix_label ${log_suffix_label}"
  fi

  if [[ "$exp_names" != *.txt ]]; then
    cmd="${cmd_template//\{img_path\}/$img_path}"
    cmd="${cmd//\{exp_name\}/$exp_names}"
    cmd="${cmd//\{job_name\}/$job_name}"
    echo $cmd
    eval $cmd --log_dpg
  else
    while IFS= read -r exp_name; do
      if [ -n "$exp_name" ] && ! [[ $exp_name == \#* ]]; then
        cmd="${cmd_template//\{img_path\}/$img_path}"
        cmd="${cmd//\{exp_name\}/$exp_name}"
        cmd="${cmd//\{job_name\}/$job_name}"
        eval $cmd --log_dpg
      fi
    done < "$exp_names"
    wait
  fi
fi

export PATH="$HOME/anaconda3/envs/sana/bin:$PATH"
echo DPG-Bench finally done
