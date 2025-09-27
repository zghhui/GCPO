#!/bin/bash
WORK_DIR="$(pwd)"
echo "$WORK_DIR"
ROOT_DIR="$WORK_DIR/.."


cd gcpo/src
RUN_NAME="gcpo"


QWEN_PATH="/home/bingxing2/ailab/quyichang/zgh/data/pretrain_model/models--deepseek-ai--Janus-Pro-1B"
HF_DATASET="$ROOT_DIR/data/geneval_train_metadata.jsonl"  
OUTPUT_MODEL_DIR="$ROOT_DIR/outputs/${RUN_NAME}_ckpt" 
OUTPUT_IMAGE_DIR="$ROOT_DIR/outputs/${RUN_NAME}_img" 

learning_rate=3e-6
reward="geneval"

PYTHONPATH="$ROOT_DIR/t2i-r1/src":$PYTHONPATH \
torchrun --nproc_per_node="4" \
    --nnodes=1 \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
open_r1/gcpo.py --use_vllm False \
    --deepspeed "../configs/zero3.json" \
    --output_dir $OUTPUT_MODEL_DIR \
    --model_name_or_path $QWEN_PATH \
    --dataset_name $HF_DATASET \
    --max_prompt_length 512 \
    --max_completion_length 1024 \
    --temperature 1.0 \
    --num_generations 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16  \
    --torch_dtype bfloat16 \
    --report_to swanlab \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_steps 1600 \
    --run_name $RUN_NAME \
    --save_steps 200 \
    --new_generations_image 4 \
    --image_token_num_per_image 576 \
    --cfg_weight 5 \
    --reward_funcs $reward \
    --beta 0.01 \
    --tf32 true \
    --learning_rate $learning_rate \
    --img_save_dir $OUTPUT_IMAGE_DIR
