#!/usr/bin/env bash

# Lấy các tham số dòng lệnh
CONFIG="configs/SHHB_final.py"
CHECKPOINT="PretrainedModels/SHHB_mae_5.8_mse_8.5.pth"
DATASET_ROOT="/home/tiennv/FPT/yolov9/STEERER/ProcessedData/test1_time/images"
GPUS_ID=${1:-0}               # Mặc định dùng GPU 0 nếu không truyền vào
PORT=${2:-29000}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Tính số lượng GPU từ GPUS_ID
GPU_NUM=0
for ((i=0; i<${#GPUS_ID}; i++)); do
    if [[ ${GPUS_ID:i:1} =~ [0-9] ]]; then
        ((GPU_NUM++))
    fi
done

# Sử dụng CPU (theo cấu hình hiện tại)
# export CUDA_VISIBLE_DEVICES=""
export CUDA_VISIBLE_DEVICES=${GPUS_ID}

# Chạy script test
python tools/test_loc.py \
    --cfg "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --cfg-options dataset.root="$DATASET_ROOT"