#!/usr/bin/env bash

export PYTHONPATH=$(pwd)
python tools/export2onnx.py \
    --cfg configs/SHHB_final.py \
    --checkpoint PretrainedModels/SHHB_mae_5.8_mse_8.5.pth \
    --output /home/tiennv/FPT/yolov9/STEERER/weights_onnx/model.onnx \