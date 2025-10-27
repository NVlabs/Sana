#!/bin/bash
set -e

work_dir=output/debug_video
np=1


if [[ $1 == *.yaml ]]; then
    config=$1
    shift
else
    config="configs/sana_video_config/256ms/Sana_2000M_256px_AdamW_fsdp.yaml.yaml"
    echo "Only support .yaml files, but get $1. Set to --config_path=$config"
fi

export DISABLE_XFORMERS=1
export DEBUG_MODE=1

cmd="TRITON_PRINT_AUTOTUNING=1 \
    torchrun --nproc_per_node=$np --master_port=$((RANDOM % 10000 + 20000))  \
        train_video_scripts/train_video_ivjoint.py \
        --config_path=$config \
        --work_dir=$work_dir \
        --train.log_interval=1 \
        --name=tmp \
        --resume_from=latest \
        --report_to=tensorboard \
        --train.num_workers=0 \
        --train.visualize=False \
        --debug=true \
        $@"

echo $cmd
eval $cmd
