#!/bin/bash

export PATH="$HOME/anaconda3/envs/geneval/bin:$PATH"
# ===================== hyper =================
geneval=true

np=8    # number of GPU to use
py=tools/metrics/geneval/evaluation/evaluate_images.py
default_sample_nums=553
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
        --log_geneval=*)
        log_geneval="${arg#*=}"
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
samples_per_gpu=$((sample_nums / np))
log_suffix_label=${suffix_label:-$default_log_suffix_label}
log_geneval=${log_geneval:-true}
tracker_project_name=${tracker_project_name:-"t2i-evit-baseline"}
echo "sample_nums: $sample_nums"
echo "log_geneval: $log_geneval"
echo "wandb_project_name: $tracker_project_name"

mask2former_path=output/pretrained_models/geneval
if [ ! -d "$mask2former_path" ]; then
  echo "Model path does not exist. Running download_models.sh..."
  bash tools/metrics/geneval/evaluation/download_models.sh $mask2former_path
fi

cmd_template="python $py --img_path {img_path} --exp_name {exp_name} \
            --model-path  $mask2former_path \
            --report_to $report_to --name {job_name} --tracker_project_name $tracker_project_name"

if [ "$geneval" = true ]; then
  # =============== compute GenEval from json ==================
  echo "==================== computing geneval ===================="

  if [[ "$exp_names" != *.txt ]]; then
    exp_name=$(basename "$exp_names")
    cmd="${cmd_template//\{img_path\}/$img_path}"
    cmd="${cmd//\{exp_name\}/$exp_name}"
    cmd="${cmd//\{job_name\}/$job_name}"
    cmd="${cmd//\{gpu_id\}/0}"
    eval CUDA_VISIBLE_DEVICES=0 $cmd >> "${img_path}/${exp_name}_geneval_result.txt" 2>&1
    cat "${img_path}/${exp_name}_geneval_result.txt"
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
        exp_name=$(basename "$exp_name")
        cmd="${cmd_template//\{img_path\}/$img_path}"
        cmd="${cmd//\{exp_name\}/$exp_name}"
        cmd="${cmd//\{job_name\}/$job_name}"
        echo "Running on GPU $gpu_id: $cmd"
        eval CUDA_VISIBLE_DEVICES=$gpu_id $cmd >> "${img_path}/${exp_name}_geneval_result.txt" 2>&1 &

        gpu_id=$(( (gpu_id + 1) % 8 ))
        job_count=$((job_count + 1))

        if [ $job_count -ge $max_parallel_jobs ]; then
          wait
          job_count=0
        fi
      fi
    done < "$exp_names"
    wait
    # show the results
    while IFS= read -r exp_name; do
      if [ -n "$exp_name" ] && ! [[ $exp_name == \#* ]]; then
        cat "${img_path}/${exp_name}_geneval_result.txt"
      fi
    done < "$exp_names"
  fi
fi

# =============== log GenEval result online after the above result saving ==================
if [ "$log_geneval" = true ] && [ "$geneval" = true ]; then
  echo "==================== logging onto $report_to ===================="

  if [ -n "${log_suffix_label}" ]; then
    echo "log_suffix_label: $log_suffix_label"
    cmd_template="${cmd_template} --suffix_label ${log_suffix_label}"
  fi

  if [[ "$exp_names" != *.txt ]]; then
    exp_name=$(basename "$exp_names")
    cmd="${cmd_template//\{img_path\}/$img_path}"
    cmd="${cmd//\{exp_name\}/$exp_name}"
    cmd="${cmd//\{job_name\}/$job_name}"
    echo $cmd
    eval $cmd --log_geneval
  else
    while IFS= read -r exp_name; do
      if [ -n "$exp_name" ] && ! [[ $exp_name == \#* ]]; then
        exp_name=$(basename "$exp_name")
        cmd="${cmd_template//\{img_path\}/$img_path}"
        cmd="${cmd//\{exp_name\}/$exp_name}"
        cmd="${cmd//\{job_name\}/$job_name}"
        eval $cmd --log_geneval
      fi
    done < "$exp_names"
    wait
  fi
fi

export PATH="$HOME/anaconda3/envs/sana/bin:$PATH"
echo GenEval finally done
