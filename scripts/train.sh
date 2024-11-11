#!/bin/bash

# Print debug information
echo "Current directory: $(pwd)"
echo "Python path: $PYTHONPATH"

# Training configuration
OUTPUT_PATH="output"
CONFIG_PATH="configs/model_config.yaml"

echo "Running with arguments:"
echo "  --config_path: $CONFIG_PATH"
echo "  --output_file_path: $OUTPUT_PATH"

# Create necessary directories
mkdir -p logs
mkdir -p output
mkdir -p cache

# Run the actual training
python src/train/train2.py \
    --config_path "$CONFIG_PATH" \
    --output_file_path "$OUTPUT_PATH" \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log