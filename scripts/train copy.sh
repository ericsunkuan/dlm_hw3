#!/bin/bash

# Print debug information
echo "Current directory: $(pwd)"
echo "Python path: $PYTHONPATH"

# Training configuration
DICT_PATH="cache/basic_event_dictionary.pkl"
OUTPUT_PATH="output"

echo "Running with arguments:"
echo "  --dict_path: $DICT_PATH"
echo "  --output_file_path: $OUTPUT_PATH"

# Run the actual training
python src/train/train2.py \
    --dict_path "$DICT_PATH" \
    --output_file_path "$OUTPUT_PATH" \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log