#! /bin/bash
set -e

sana_dir=$1
number_of_files=$2
pick_number=$3
# calculate number of GPU to use in this machine
num_gpu=$(nvidia-smi -L | wc -l)
echo "sana_dir: $sana_dir, number_of_files: $number_of_files, pick_number: $pick_number, num_gpu: $num_gpu"
# start idx iterate from 0 * (552//8), 1 * (552//8), 2 * (552//8), 3 * (552//8), 4 * (552//8), 5 * (552//8), 6 * (552//8), 7 * (552//8)
# end idx iterate from 1 * (552//8), 2 * (552//8), 3 * (552//8), 4 * (552//8), 5 * (552//8), 6 * (552//8), 7 * (552//8), 552
for idx in $(seq 0 $((num_gpu - 1))); do
    start_idx=$((idx * (552 / num_gpu)))
    end_idx=$((start_idx + 552 / num_gpu))
    if [ $idx -eq $((num_gpu - 1)) ]; then
        end_idx=552
    fi

    echo "CUDA_VISIBLE_DEVICES=$idx python tools/inference_scaling/nvila_sana_pick.py --start_idx $start_idx --end_idx $end_idx --base_dir $sana_dir --number_of_files $number_of_files --pick_number $pick_number &"
    CUDA_VISIBLE_DEVICES=$idx python tools/inference_scaling/nvila_sana_pick.py --start_idx $start_idx --end_idx $end_idx --base_dir $sana_dir --number_of_files $number_of_files --pick_number $pick_number &
done
wait
