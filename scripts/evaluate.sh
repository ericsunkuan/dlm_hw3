# scripts/evaluate.sh
#!/bin/bash

# Set environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Paths
DICT_PATH="cache/dictionary.pkl"  # The dictionary will be saved here during training
OUTPUT_PATH="outputs/task1/midi"  # Path to generated MIDI files

# Run evaluation
python eval_metrics.py \
    --dict_path $DICT_PATH \
    --output_file_path $OUTPUT_PATH