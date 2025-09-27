#!/bin/bash

RANDOM_PORT=$(shuf -i 10000-65535 -n 1)
echo "Randomly selected port: $RANDOM_PORT"

ROOT_DIR="$(pwd)"
echo "$ROOT_DIR"
MASTER_PORT=12345

data_path="$ROOT_DIR/data/train_data.json"
vq_model_ckpt="vq_ds16_t2i.pt"
hps_model_path="HPS_v2.1_compressed.pt"
text_tokenizer="models--google--flan-t5-xl"

accelerate launch --config_file simpar/configs/accelerate_configs/zero2.yaml \
    --main_process_port ${MASTER_PORT} \
    --num_machines 1 \
    --num_processes 4 \
    --machine_rank 0 \
    simpar/train/llamaGen_trainer_gcpo_hps.py \
        --config simpar/configs/config_grpo_hps.yaml \
        --data_path $data_path \
        --vq_model_ckpt $vq_model_ckpt \
        --hps_model_path $hps_model_path \
        --text_tokenizer $text_tokenizer \
        --dataset_name hps-data \
        --output_dir save\
        --image_size 256\
        --downscale_factor 16