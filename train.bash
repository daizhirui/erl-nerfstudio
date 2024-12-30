#!/usr/bin/bash

set -e
set -x

#SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATA_DIR=/home/daizhirui/Data/erl_neural_sddf/datasets
OUTPUT_DIR=/home/daizhirui/results/erl_neural_sddf/baselines/
COMMON_ARGS="--pipeline.model.camera-optimizer.mode off --vis tensorboard"

for dataset in $(ls $DATA_DIR); do
    if [[ $dataset != *"rgbd" ]]; then
            continue
    fi
    echo $dataset

    for model in "nerfacto" "depth-nerfacto"; do
        ns-train $model --data $DATA_DIR/$dataset/scans/train \
            --output-dir $OUTPUT_DIR/$model/$dataset \
            $COMMON_ARGS
    done

done
