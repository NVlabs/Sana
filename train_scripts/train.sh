#/bin/bash
set -e

work_dir=output/debug
np=2


if [[ $1 == *.yaml ]]; then
    config=$1
    shift
else
    config="configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml"
    # config="configs/sana1-5_config/1024ms/Sana_1600M_1024px_AdamW_fsdp.yaml"      FSDP config file
    echo "Only support .yaml files, but get $1. Set to --config_path=$config"
fi

TRITON_PRINT_AUTOTUNING=1 \
    torchrun --nproc_per_node=$np --master_port=15432 \
        train_scripts/train.py \
        --config_path=$config \
        --work_dir=$work_dir \
        --name=tmp \
        --resume_from=latest \
        --report_to=tensorboard \
        --debug=true \
        "$@"
