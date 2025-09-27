import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import os
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional

import transformers
import sys
sys.path.insert(0, '/home/supervisor/data/ZYZ/zgh/AIGC/AR/SimpleAR')

from ddp_distributed import init_distributed_mode
from simpar.model.tokenizer.cosmos_tokenizer.networks import TokenizerConfigs
from simpar.model.tokenizer.cosmos_tokenizer.video_lib import CausalVideoTokenizer as CosmosTokenizer
from simpar.train.t2i_data import T2IDataset

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)

    gen_image_folder: Optional[str] = field(default=None)
    gen_data_path: str = field(default=None)

    gen_video_folder: Optional[str] = field(default=None)
    gen_video_path: str = field(default=None)
    sequence_length: int = field(default=17)

    gen_resolution: Optional[int] = field(default=None)

    mixture_ratio: str = field(default=None)
    p_drop_cond: float = field(default=0.0)

    vq_model_ckpt: str = field(default="/path_to_tokenizer/Cosmos-1.0-Tokenizer-DV8x16x16")
    dataset_name: str = field(default="laion")
    dataset_type: str = field(default="image")
    code_path: str = field(default="./debug")
    
    crop_type: str = field(default="centercrop")



def gather_meta(meta):
    """Gathers metadata across all DDP ranks."""
    world_size = dist.get_world_size()
    gathered_meta = [None] * world_size  # Placeholder for all ranks' data
    dist.all_gather_object(gathered_meta, meta)  # Gather lists of dictionaries
    merged_meta = sum(gathered_meta, [])  # Flatten the list
    return merged_meta

def main(data_args):
    if data_args.dataset_type == "image":
        code_dir = f"{data_args.code_path}/{data_args.dataset_name}/{data_args.gen_resolution}_codes"
        label_dir = f"{data_args.code_path}/{data_args.dataset_name}/{data_args.gen_resolution}_labels"
    else:
        code_dir = f"{data_args.code_path}/{data_args.dataset_name}/{data_args.gen_resolution}_{data_args.sequence_length}_codes"
        label_dir = f"{data_args.code_path}/{data_args.dataset_name}/{data_args.gen_resolution}_{data_args.sequence_length}_labels"

    os.makedirs(code_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    tokenizer_config = TokenizerConfigs["DV"].value
    tokenizer_config.update(dict(spatial_compression=16, temporal_compression=8))
    model = CosmosTokenizer(checkpoint_enc=f"{data_args.vq_model_ckpt}/encoder.jit", checkpoint_dec=f"{data_args.vq_model_ckpt}/decoder.jit", tokenizer_config=tokenizer_config)
    
    model.eval()
    model.requires_grad_(False)
    

    init_distributed_mode(data_args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = 0 * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    model.to(device)
    dataset = T2IDataset(data_path=data_args.gen_data_path, tokenizer=None, image_size=data_args.gen_resolution, aug="centercrop", data_args=data_args)
    if dist.get_rank() == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
        print(f"Extracting image tokens for {len(dataset)} samples.")
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=1, # important!
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    total = 0
    for data in tqdm(loader):
        image = data["image"][0]
        raw_caption = data["caption"][0].strip().replace("\n", " ")

        image_tensor, _, _ = image
        image_tensor = image_tensor.to(device).to(torch.bfloat16)
            
        if image_tensor.ndim == 5:
            image_tensor = image_tensor.permute(0, 2, 1, 3, 4).contiguous()
        else:
            image_tensor = image_tensor.unsqueeze(2)

        with torch.no_grad():
            (image_token, _) = model.encode(image_tensor)
        
        x = image_token.detach().cpu().numpy()
        train_steps = total + rank
        
        code_path = f"{code_dir}/{train_steps}.npy"
        label_path = f"{label_dir}/{train_steps}.npy"
        
        np.save(code_path, x)
        with open(label_path, "w") as f:
            f.write(raw_caption)
        
        
        total += dist.get_world_size()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = transformers.HfArgumentParser([DataArguments])
    data_args = parser.parse_args_into_dataclasses()[0]
    data_args.gen_image_folder = ""

    main(data_args)