#!/bin/bash
set -e

CONFIG_FILE="/home/tiennv/FPT/yolov9/STEERER/configs/SHHB_final.py" 
ONNX_MODEL="/home/tiennv/FPT/yolov9/STEERER/weights_onnx/model.onnx" 
INPUT_DIR="/home/tiennv/FPT/yolov9/STEERER/ProcessedData/5_img/5_img/images"  
OUTPUT_DIR="/home/tiennv/FPT/yolov9/STEERER/onnx_output"       
INPUT_SHAPE="3 1024 2048"              
LOC_THRESHOLD="0.1"                  

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE does not exist"
    exit 1
fi

if [ ! -f "$ONNX_MODEL" ]; then
    echo "Error: ONNX model file $ONNX_MODEL does not exist"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR does not exist"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Starting ONNX inference..."
python tools/inference4onnx.py \
    --cfg "$CONFIG_FILE" \
    --onnx-model "$ONNX_MODEL" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --input-shape $INPUT_SHAPE \
    --device cuda