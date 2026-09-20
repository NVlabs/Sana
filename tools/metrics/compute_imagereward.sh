#!/bin/bash

# ===================== hyper =================
imagereward=true

py=tools/metrics/image_reward/compute_image_reward.py
default_sample_nums=100
report_to=wandb
default_log_suffix_label=''

# parser
img_path=$1
exp_names=$2
job_name=$(basename $(dirname "$img_path"))

for arg in "$@"
do
    case $arg in
        --sample_nums=*)
        sample_nums="${arg#*=}"
        shift
        ;;
        --suffix_label=*)
        suffix_label="${arg#*=}"
        shift
        ;;
        --log_image_reward=*)
        log_image_reward="${arg#*=}"
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
        *)
        ;;
    esac
done

sample_nums=${sample_nums:-$default_sample_nums}
tracker_pattern=${tracker_pattern:-"epoch_step"}
log_suffix_label=${suffix_label:-$default_log_suffix_label}
log_image_reward=${log_image_reward:-true}
tracker_project_name=${tracker_project_name:-"t2i-evit-baseline"}
echo "img_size: $img_size"
echo "sample_nums: $sample_nums"
echo "log_image_reward: log_image_reward"
echo "log_suffix_label: $log_suffix_label"
echo "tracker_pattern: $tracker_pattern"
echo "wandb_project_name: $tracker_project_name"
JSON_PATH="tools/metrics/image_reward/benchmark-prompts-dict.json"

if [ "$imagereward" = true ]; then
  # =============== compute image-reward from two jsons ==================
  echo "==================== computing image-reward ===================="
  cmd_template="python $py --json_path $JSON_PATH \
              --exp_name {exp_name} --txt_path {img_path} --img_path {img_path} --sample_nums $sample_nums \
              --report_to $report_to --name {job_name} --gpu_id {gpu_id} --tracker_pattern $tracker_pattern \
              --tracker_project_name $tracker_project_name"

  if [[ "$exp_names" != *.txt ]]; then
    cmd="${cmd_template//\{img_path\}/$img_path}"
    cmd="${cmd//\{exp_name\}/$exp_names}"
    cmd="${cmd//\{job_name\}/$job_name}"
    cmd="${cmd//\{gpu_id\}/0}"
    eval CUDA_VISIBLE_DEVICES=0 $cmd
  else

    if [ ! -f "$exp_names" ]; then
      echo "Model paths file not found: $exp_names"
      exit 1
    fi

    gpu_id=0
    max_parallel_jobs=8
    job_count=0
    echo "" >> "$exp_names"   # add a new line to the file avoid skipping last line dir

    while IFS= read -r exp_name; do
      echo $exp_name
      if [ -n "$exp_name" ] && ! [[ $exp_name == \#* ]]; then
        cmd="${cmd_template//\{img_path\}/$img_path}"
        cmd="${cmd//\{exp_name\}/$exp_name}"
        cmd="${cmd//\{job_name\}/$job_name}"
        cmd="${cmd//\{gpu_id\}/$gpu_id}"
        echo "Running on GPU $gpu_id: $cmd"
        eval CUDA_VISIBLE_DEVICES=$gpu_id $cmd &

        gpu_id=$(( (gpu_id + 1) % 8 ))
        job_count=$((job_count + 1))

        if [ $job_count -ge $max_parallel_jobs ]; then
          wait
          job_count=0
        fi
      fi
    done < "$exp_names"
    wait
  fi
fi

# =============== log image-reward result online after the above result saving ==================
if [ "$log_image_reward" = true ] && [ "$imagereward" = true ]; then
    echo "==================== logging onto $report_to ===================="

  if [ -n "${log_suffix_label}" ]; then
    cmd_template="${cmd_template} --suffix_label ${log_suffix_label}"
  fi

  if [[ "$exp_names" != *.txt ]]; then
    cmd="${cmd_template//\{img_path\}/$img_path}"
    cmd="${cmd//\{exp_name\}/$exp_names}"
    cmd="${cmd//\{job_name\}/$job_name}"
    cmd="${cmd//\{gpu_id\}/0}"
    echo $cmd
    eval $cmd --log_image_reward
  else
    while IFS= read -r exp_name; do
      if [ -n "$exp_name" ] && ! [[ $exp_name == \#* ]]; then
        cmd="${cmd_template//\{img_path\}/$img_path}"
        cmd="${cmd//\{exp_name\}/$exp_name}"
        cmd="${cmd//\{job_name\}/$job_name}"
        cmd="${cmd//\{gpu_id\}/0}"
        eval $cmd --log_image_reward
      fi
    done < "$exp_names"
    wait
  fi
fi

echo image-reward finally done
