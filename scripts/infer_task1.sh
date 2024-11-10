#!/bin/bash

# Set environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Configuration
# /home/paperspace/dlmusic/hw3/logs/20241109_133229/checkpoints/checkpoint_epoch8_step0.pt
CHECKPOINT_PATH="logs/20241109_133229/checkpoints/checkpoint_epoch8_step0.pt"  # Replace with your actual checkpoint path
CONFIG_PATH="configs/model_config.yaml"
OUTPUT_DIR="outputs"

# Run generation for Task 1
python src/generate/generate.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --config "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --task 1