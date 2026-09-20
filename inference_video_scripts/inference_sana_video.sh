#!/bin/bash
set -e

inference_script=inference_video_scripts/inference_sana_video.py
config=""
model_path=""
np=8
negative_prompt=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --config=*)
      config="${1#*=}"
      shift
      ;;
    --config)
      config="$2"
      shift 2
      ;;
    --model_path=*)
      model_path="${1#*=}"
      shift
      ;;
    --model_path)
      model_path="$2"
      shift 2
      ;;
    --inference_script=*)
      inference_script="${1#*=}"
      shift
      ;;
    --inference_script)
      inference_script="$2"
      shift 2
      ;;
    --np=*)
      np="${1#*=}"
      shift
      ;;
    --np)
      np="$2"
      shift 2
      ;;
    --negative_prompt=*)
      negative_prompt="${1#*=}"
      shift
      ;;
    --negative_prompt)
      negative_prompt="$2"
      shift 2
      ;;
    *)
      other_args+=("$1")
      shift
      ;;
  esac
done

cmd=(
  accelerate launch --num_processes="$np" --num_machines=1 --mixed_precision=bf16 --main_process_port="$RANDOM"
  "$inference_script"
  --config="$config"
  --model_path="$model_path"
  --txt_file=asset/samples/video_prompts_samples.txt
  --dataset=video_samples
)

if [[ -n "$negative_prompt" ]]; then
  cmd+=(--negative_prompt="$negative_prompt")
fi

if [[ ${#other_args[@]} -gt 0 ]]; then
  cmd+=("${other_args[@]}")
fi

printf -v cmd_str '%q ' "${cmd[@]}"
echo "$cmd_str"

"${cmd[@]}"
