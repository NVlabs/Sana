#/bin/bash
set -e

work_dir=output/debug_sCM_ladd
np=2


if [[ $1 == *.yaml ]]; then
    config=$1
    shift
else
    config="configs/sana_sprint_config/1024ms/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml"
    echo "Only support .yaml files, but get $1. Set to --config_path=$config"
fi

cmd="TRITON_PRINT_AUTOTUNING=1 \
    torchrun --nproc_per_node=$np --master_port=$((RANDOM % 10000 + 20000)) \
        train_scripts/train_scm_ladd.py \
        --config_path=$config \
        --work_dir=$work_dir \
        --name=tmp \
        --resume_from=latest \
        --report_to=tensorboard \
        --debug=true \
        $@"

echo $cmd
eval $cmd
