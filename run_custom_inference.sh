#!/bin/bash

# Custom Prompt Inference Script for DeepSeek-VL2
# Process images in error_images directory without ground truth

source /data/isaackang/anaconda3/etc/profile.d/conda.sh
conda activate deepseekvl
cd /data/isaackang/Others/DeepSeek-VL2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4

PROMPT="What is the main word in the image? Why did you think so? Are there any characters that are partly occluded or cropped at the edges?"

# Run inference
python custom_prompt_inference.py \
    --image_dir error_images \
    --model_name "deepseek-ai/deepseek-vl2-tiny" \
    --device auto \
    --output_file "custom_inference_results.txt" \
    --prompt "$PROMPT" \
    --max_new_tokens 1000 \
    "$@"

echo "Custom Prompt Inference completed!"

