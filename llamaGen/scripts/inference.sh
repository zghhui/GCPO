MODEL_PATH="Your checkpoint"
vq_model_ckpt="vq_ds16_t2i.pt"
OUTPUT_BASE="eval/samples"

# 随机生成一个端口号（假设端口范围是 10000 到 65535）
RANDOM_PORT=$(shuf -i 10000-65535 -n 1)
echo "Randomly selected port: $RANDOM_PORT"

# 启动 accelerate
echo "=======Starting inference======="

PROMPT_FILE="data/test.txt"
accelerate launch --num-processes 1 --main_process_port $RANDOM_PORT simpar/sample/sample_t2i.py \
    --model_path "$MODEL_PATH" \
    --vq_model_ckpt $vq_model_ckpt \
    --prompt_file "$PROMPT_FILE" \
    --output_base "$OUTPUT_BASE" \
    --latent_dim 256 \
    --temperature 1.0 \
    --top_k 0 \
    --top_p 1.0 \
    --cfg 7.5