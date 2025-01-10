#!/usr/bin/bash

set -e

MODELS="$@"
if [ -z "$MODELS" ]; then
    MODELS="nerfacto depth-nerfacto"
fi
SKIP_EXISTING=${SKIP_EXISTING:-0}  # default to not skip existing results

#SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATA_DIR=/home/daizhirui/Data/erl_neural_sddf/datasets
OUTPUT_DIR=/home/daizhirui/results/erl_neural_sddf
MODEL_ARGS="--pipeline.model.camera-optimizer.mode off"
OTHER_ARGS="--vis tensorboard --max-num-iterations 50000"
DATA_ARGS="nerfstudio-data --depth-unit-scale-factor 1.0 --eval-mode filename"  # dataset is in meters not millimeters, which is the default of NeRF-Studio

for dataset in $(ls $DATA_DIR); do
    if [[ $dataset != *"rgbd" ]]; then
            continue
    fi
    echo $dataset

    for model in $MODELS; do
        if [ ! -d $OUTPUT_DIR/$model/$dataset ] || [ $SKIP_EXISTING -eq 0 ]; then
            set -x
            ns-train $model --data $DATA_DIR/$dataset \
                --output-dir $OUTPUT_DIR/$model \
                $MODEL_ARGS $OTHER_ARGS $DATA_ARGS
            set +x
        else
            echo "Skipping $model/$dataset"
        fi
        if [ ! -d $OUTPUT_DIR/$model-normals/$dataset ] || [ $SKIP_EXISTING -eq 0 ]; then
            set -x
            ns-train $model --data $DATA_DIR/$dataset \
                --output-dir $OUTPUT_DIR/$model-normals \
                --pipeline.model.predict-normals True \
                $MODEL_ARGS $OTHER_ARGS $DATA_ARGS
            set +x
        else
            echo "Skipping $model-normals/$dataset"
        fi
    done
done
