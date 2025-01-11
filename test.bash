#!/usr/bin/bash

set -e
#set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATA_DIR=/home/daizhirui/Data/erl_neural_sddf/datasets
OUTPUT_DIR=/home/daizhirui/results/erl_neural_sddf

for dataset in $(ls $DATA_DIR); do
    if [[ $dataset != *"rgbd" ]]; then
        continue
    fi
    echo $dataset

    for model in "nerfacto" "depth-nerfacto"; do
        MODEL_DIR=$OUTPUT_DIR/$model/$dataset/$model
        for exp_folder in $(ls $MODEL_DIR); do
            n_ckpt=$(ls $MODEL_DIR/$exp_folder/nerfstudio_models | wc -l)
            if [[ $n_ckpt -eq 0 ]]; then  # Skip if no checkpoint
                echo "No checkpoint found in $MODEL_DIR/$exp_folder/nerfstudio_models"
                continue
            fi
            echo $OUTPUT_DIR/$model/$dataset
            mkdir -p $OUTPUT_DIR/$model/$dataset/test
            set -x
            PYTHONPATH=$(pwd) python3 nerfstudio/scripts/render.py dataset --load-config $MODEL_DIR/$exp_folder/config.yml \
                --output-path $OUTPUT_DIR/$model/$dataset \
                --split test \
                --rendered-output-names rgb depth \
                --colormap-options.colormap jet | tee $OUTPUT_DIR/$model/$dataset/test/log.txt
            set +x
#            break   # each model only has one exp folder
        done

        MODEL_DIR=$OUTPUT_DIR/$model-normals/$dataset/$model
        for exp_folder in $(ls $MODEL_DIR); do
            n_ckpt=$(ls $MODEL_DIR/$exp_folder/nerfstudio_models | wc -l)
            if [[ $n_ckpt -eq 0 ]]; then  # Skip if no checkpoint
                echo "No checkpoint found in $MODEL_DIR/$exp_folder/nerfstudio_models"
                continue
            fi
            echo $OUTPUT_DIR/$model-normals/$dataset
            mkdir -p $OUTPUT_DIR/$model-normals/$dataset/test
            set -x
            PYTHONPATH=$SCRIPT_DIR python3 $SCRIPT_DIR/nerfstudio/scripts/render.py dataset \
                --load-config $MODEL_DIR/$exp_folder/config.yml \
                --output-path $OUTPUT_DIR/$model-normals/$dataset \
                --split test \
                --rendered-output-names rgb depth normals \
                --colormap-options.colormap jet | tee $OUTPUT_DIR/$model-normals/$dataset/test/log.txt
            set +x
#            break   # each model only has one exp folder
        done
    done
done
