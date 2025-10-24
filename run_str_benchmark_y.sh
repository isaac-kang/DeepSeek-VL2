#!/bin/bash
# Script to evaluate STR LMDB benchmarks with DeepSeek-VL2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export STR_DATA_DIR=/data/isaackang/data/STR/english_case-sensitive/lmdb/evaluation

# Default: process all samples per dataset with optimized prompt
# Options:
#   --num_samples N           : Number of samples per dataset (default: -1 for all)
#   --batch_size N            : Batch size for inference (default: 1)
#   --datasets "CUTE80,SVT"   : Specific datasets to evaluate
#   --prompt "text"           : Custom prompt
#   --case-sensitive false    : Case-insensitive matching (default: false)
#   --ignore-punctuation true : Ignore punctuation (default: true)
#   --ignore-space true       : Ignore spaces (default: true)
#   --model MODEL             : Different model
#   --results_dir DIR         : Results directory (default: str_benchmark_results)
#   --chunk_size N            : Chunk size for incremental prefilling (default: -1)

    # --datasets "CUTE80,SVT,SVTP,IC13_857,IC15_1811,IIIT5k_3000" \
python str_evaluation.py \
    --datasets "CUTE80" \
    --model_name "deepseek-ai/deepseek-vl2-small" \
    --max_samples 10 \
    --batch_size 1 \
    --device auto \
    --prompt "What is the main word in the image? Output only the text." \
    --case-sensitive false \
    --ignore-punctuation true \
    --ignore-spaces true \
    --chunk_size -1 \
    --results_dir "str_benchmark_results" \
    "$@"
