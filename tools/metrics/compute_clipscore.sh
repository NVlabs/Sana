#!/bin/bash

# ===================== hyper =================
clip_score=true

py=tools/metrics/clip-score/clip_score.py
default_sample_nums=30000   # 10000, 30000
report_to=wandb
default_log_suffix_label=''

# parser
img_path=$1
exp_names=$2
job_name=$(basename $(dirname "$img_path"))
#job_name=online_monitor_debug

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
        --log_clip_score=*)
        log_clip_score="${arg#*=}"
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
log_clip_score=${log_clip_score:-true}
tracker_project_name=${tracker_project_name:-"t2i-evit-baseline"}
echo "sample_nums: $sample_nums"
echo "log_clip_score: $log_clip_score"
echo "tracker_pattern: $tracker_pattern"
echo "wandb_project_name: $tracker_project_name"
IMG_PATH="data/test/PG-eval-data/MJHQ-30K/meta_data.json"
TXT_PATH="data/test/PG-eval-data/MJHQ-30K/meta_data.json"

#CUDA_VISIBLE_DEVICES=0 python  --real_path $IMG_PATH --fake_path $TXT_PATH

if [ "$clip_score" = true ]; then
  # =============== compute clip-score from two jsons ==================
  echo "==================== computing clip-score ===================="
  cmd_template="python $py --real_path $IMG_PATH --fake_path $TXT_PATH \
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

# =============== log clip-score result online after the above result saving ==================
if [ "$log_clip_score" = true ] && [ "$clip_score" = true ]; then
    echo "==================== logging onto $report_to ===================="

  if [ -n "${log_suffix_label}" ]; then
    echo "log_suffix_label: $log_suffix_label"
    cmd_template="${cmd_template} --suffix_label ${log_suffix_label}"
  fi

  if [[ "$exp_names" != *.txt ]]; then
    cmd="${cmd_template//\{img_path\}/$img_path}"
    cmd="${cmd//\{exp_name\}/$exp_names}"
    cmd="${cmd//\{job_name\}/$job_name}"
    cmd="${cmd//\{gpu_id\}/0}"
    echo $cmd
    eval $cmd --log_clip_score
  else
    while IFS= read -r exp_name; do
      if [ -n "$exp_name" ] && ! [[ $exp_name == \#* ]]; then
        cmd="${cmd_template//\{img_path\}/$img_path}"
        cmd="${cmd//\{exp_name\}/$exp_name}"
        cmd="${cmd//\{job_name\}/$job_name}"
        cmd="${cmd//\{gpu_id\}/0}"
        eval $cmd --log_clip_score
      fi
    done < "$exp_names"
    wait
  fi
fi

echo clip-score finally done
