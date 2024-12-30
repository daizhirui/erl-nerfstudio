#!/usr/bin/bash

set -e
#set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATA_DIR=/home/daizhirui/Data/erl_neural_sddf/datasets

for dataset in $(ls $DATA_DIR); do
    if [[ $dataset != *"rgbd" ]]; then
        continue
    fi
    echo $dataset
    python3 $SCRIPT_DIR/convert_to_nerf.py \
        --base_path $DATA_DIR/$dataset/scans/train --output_file $DATA_DIR/$dataset/scans/train/transforms.json
    python3 $SCRIPT_DIR/convert_to_nerf.py \
        --base_path $DATA_DIR/$dataset/scans/test --output_file $DATA_DIR/$dataset/scans/test/transforms.json
done
