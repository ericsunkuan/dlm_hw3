#!/bin/bash

# Set environment variables for better GPU performance
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Training configuration
CONFIG_PATH="configs/model_config.yaml"
DATA_DIR="src/data/Pop1K7"  # Replace with your dataset path
DICT_PATH="cache/basic_event_dictionary.pkl"
OUTPUT_PATH="output"

# Update data directory in config
sed -i "s|data_dir:.*|data_dir: \"$DATA_DIR\"|" $CONFIG_PATH

# Create output directory
mkdir -p $OUTPUT_PATH

# Run training with arguments in the correct order
python src/train/train.py \
    --dict_path $DICT_PATH \
    --output_file_path $OUTPUT_PATH \
    --config $CONFIG_PATH \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log