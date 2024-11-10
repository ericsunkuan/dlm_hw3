# scripts/config.sh
#!/bin/bash

# Training configurations
export BATCH_SIZE=16
export NUM_WORKERS=4
export LEARNING_RATE=0.0001
export NUM_EPOCHS=100
export SEQUENCE_LENGTH=512
export MODEL_DIM=512
export NUM_LAYERS=12
export NUM_HEADS=8

# Update config file
cat > configs/model_config.yaml << EOF
model:
  n_layer: $NUM_LAYERS
  n_head: $NUM_HEADS
  d_model: $MODEL_DIM
  d_head: 64
  d_inner: 2048
  dropout: 0.1
  dropatt: 0.0
  mem_len: $SEQUENCE_LENGTH
  ext_len: 0
  tgt_len: $SEQUENCE_LENGTH

data:
  data_dir: "path/to/pop_dataset"
  vocab_size: 512
  sequence_length: $SEQUENCE_LENGTH
  batch_size: $BATCH_SIZE
  num_workers: $NUM_WORKERS

training:
  epochs: $NUM_EPOCHS
  learning_rate: $LEARNING_RATE
  warmup_steps: 1000
  gradient_clip_val: 1.0
  fp16: true
  log_every_n_steps: 100
  eval_every_n_steps: 1000
  save_every_n_steps: 5000

generation:
  temperatures: [0.8, 1.0, 1.2]
  top_k: [20, 40, 60]
  num_samples: 20
  max_length: 2048
EOF