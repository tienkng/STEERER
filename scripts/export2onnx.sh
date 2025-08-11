#!/usr/bin/env bash

export PYTHONPATH=$(pwd)
python tools/export2onnx.py \
    --model-cfg 'configs/SHHB_final.py' \
    --checkpoint 'weights/SHHB_mae_5.8_mse_8.5.pth' \
    --batch-size 3