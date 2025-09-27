import os
import sys
import argparse
from accelerate import Accelerator
import torch
from torchvision.utils import save_image


# 获取项目根目录（假设llamaGen_trainer_grpo.py在simpar/train/下，往上两级）
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.insert(0, project_root)
from simpar.model.llama_model import LlamaForCausalLM
from simpar.model.llamagen.vq_model import VQ_16

def parse_args():
    parser = argparse.ArgumentParser(description="Sample Inference with LlamaForCausalLM")
    parser.add_argument('--model_path', type=str, default='', help="Checkpoint Path")
    parser.add_argument('--vq_model_ckpt', type=str, default='', help="VQ Model Checkpoint Path")
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'], help="Data type")
    parser.add_argument('--prompt_file', type=str, default='', help="Path to the prompt file")
    parser.add_argument('--output_base', type=str, default='', help="Base directory for output")
    parser.add_argument('--latent_dim', type=int, default=256, help="Latent dimension")
    parser.add_argument('--max_new_tokens', type=int, default=None, help="Maximum number of new tokens to generate")
    parser.add_argument('--temperature', type=float, default=1.0, help="Temperature for generation")
    parser.add_argument('--top_k', type=int, default=1000, help="Top-k for sampling")
    parser.add_argument('--top_p', type=float, default=1.0, help="Top-p (nucleus sampling) for generation")
    parser.add_argument('--cfg', type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for generation")

    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()

    # 1. 初始化 accelerate
    accelerator = Accelerator()
    device = accelerator.device
    rank = accelerator.process_index      # 当前进程 rank: 0,1,...
    world_size = accelerator.num_processes  # 总进程数

    # 2. 路径 & 配置
    MODEL_NAME = args.model_path
    dtype = torch.__dict__[args.dtype]  # 根据输入类型选择 dtype
    prompt_file = args.prompt_file
    output_base = args.output_base

    # 3. 读取所有 prompts，并按 rank 划分
    with open(prompt_file, "r", encoding="utf-8") as f:
        all_prompts = [l.strip() for l in f if l.strip()]
    # 每个进程处理自己分到的那部分：round-robin 切片
    prompts = all_prompts[rank::world_size]

    # 4. 加载模型到对应设备
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device=device, dtype=dtype)
    model.eval()
    
    # Load VQ model
    vq_model = VQ_16(codebook_size=16384, codebook_embed_dim=8)
    checkpoint = torch.load(args.vq_model_ckpt, weights_only=False)
    vq_model.load_state_dict(checkpoint['model'])
    vq_model = vq_model.to(device)
    vq_model.eval()

    # 5. 确保输出目录存在
    os.makedirs(output_base, exist_ok=True)

    # 6. 依次生成
    latent_dim = args.latent_dim // 16
    batch_size = args.batch_size
    num_batches = len(prompts) // batch_size + (1 if len(prompts) % batch_size > 0 else 0)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens else latent_dim ** 2
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_prompts = prompts[batch_start: batch_start + batch_size]
        real_num = len(batch_prompts)
        if real_num < batch_size:
            # 补齐空prompt
            batch_prompts = batch_prompts + [""] * (batch_size - real_num)
        params = {
            "prompt": batch_prompts,
            "max_length": max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "cfg": args.cfg,
        }
        _, _, completion_ids, _ = model.generate(**params)
        _, imgs = model.decode_ids(completion_ids, vq_model)
        # 只保存真实prompt对应的图片
        for i, (prompt, img) in enumerate(zip(batch_prompts[:real_num], imgs[:real_num])):
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in prompt)
            global_idx = (batch_start + i) * world_size + rank
            name_len = min(50, len(safe))
            out_path = os.path.join(output_base, f"{global_idx}_{safe[:name_len]}.png")
            save_image(img, out_path, normalize=True, value_range=(-1, 1))
            print(f"[rank {rank}] prompt #{batch_start + i} → saved: {out_path}")

if __name__ == "__main__":
    main()
