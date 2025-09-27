import os
import re
import json
import time
import yaml
import math
import random
from PIL import Image
from typing import Dict
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import transformers
from simpar.train.preprocess import preprocess_t2i
from simpar.utils import rank0_print


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class T2IDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, image_size=384, aug="resize", data_args=None):
        super(T2IDataset, self).__init__()
        
        self.list_data_dict = []
        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            with open(data_path, "r") as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)

        
        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")

        self.aug = aug
        if aug == "resize":
            augmentations = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        elif aug == "centercrop":
            augmentations = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        else: # any res
            augmentations = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                ]
            )
        
        self.transform = augmentations

        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = len(sample["prompt"].split())
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            # if "image" in sample or "video" in sample or self.data_args.early_mix_text:
            length_list.append(cur_len)
            
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"T2I....[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        gen_image_folder = self.data_args.gen_image_folder

        image_file = sources["image_path"]
        prompt = sources["caption"]
        if isinstance(prompt, list):
            prompt = prompt[0]
        
        image = Image.open(os.path.join(gen_image_folder, image_file)).convert("RGB")
        image_tensor = self.transform(image)
        image_gen = [(image_tensor, image.size, "image")]
        
        
        data_dict = dict(
            image=image_gen,
            data_type="gen",
            image_path=image_file,
            caption=prompt,
        )

        return data_dict

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class EvalT2IDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, image_folder: str, data_path: str, tokenizer: transformers.PreTrainedTokenizer, image_size: int, benchmark: str = "mjhq", raw_prompt=False, num_chunks=1, chunk_idx=0):
        super(EvalT2IDataset, self).__init__()

        if benchmark == "mjhq" or benchmark == "coco":
            list_data_dict = json.load(open(data_path, "r"))
        elif benchmark == "geneval":
            list_data_dict = []
            with open(data_path, "r") as f:
                for line in f:
                    list_data_dict.append(json.loads(line))
        elif benchmark == "dpg":
            list_data_dict = []
            for file in os.listdir(data_path):
                list_data_dict.append(os.path.join(data_path, file))

        self.benchmark = benchmark
        self.image_folder = image_folder
        self.tokenizer = tokenizer

        list_data_dict = get_chunk(list_data_dict, num_chunks, chunk_idx)
        self.list_data_dict = list_data_dict
        
        augmentations = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.image_size = image_size
        self.transform = augmentations
        self.raw_prompt = raw_prompt

    def __len__(self):
        return len(self.list_data_dict)


    def __getitem__(self, i):
        item = self.list_data_dict[i]

        if self.benchmark == "mjhq":
            image_file = item["image_path"]
            prompt = "The image features " + item["caption"]
            meta = item["image_path"]
        
        elif self.benchmark == "coco":
            image_file = item["image_path"]
            prompt = item["caption"] if self.raw_prompt else item["moondream_caption"]
            meta = item["image_path"]

        elif self.benchmark == "geneval":
            image_file = None
            prompt = item["prompt"] if "reprompt" not in item else item["reprompt"]
            meta = item
        
        elif self.benchmark == "dpg":
            image_file = None
            prompt = open(item).read()
            meta = {"prompt": prompt, "file_name": os.path.basename(item).replace(".txt", ".jpg")}
        
        try:
            image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
            image_tensor = self.transform(image)
        except:
            image_tensor = torch.rand(3, self.image_size, self.image_size)
        
        sources = [
            [
                {'from': 'human', 'value': '<image>'}, 
                {'from': 'gpt', 'value': prompt}
            ]
        ]
        
        
        t2i_data_dict = preprocess_t2i(sources, self.tokenizer, vtokens_shape=0, p_drop_cond=0.)
        sources2 = [
            [
                {'from': 'human', 'value': '<image>'}, 
                {'from': 'gpt', 'value': "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"}
            ]
        ]
        uncond_t2i_data_dict = preprocess_t2i(sources2, self.tokenizer, vtokens_shape=0, p_drop_cond=0.)

        return image_tensor, t2i_data_dict["input_ids"][0][0], uncond_t2i_data_dict["input_ids"][0][0], meta
    


class GRPOT2IDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(GRPOT2IDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.list_data_dict = list_data_dict
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.list_data_dict)


    def __getitem__(self, i):
        item = self.list_data_dict[i]
        sources = [
            [
                {'from': 'human', 'value': '<image>'}, 
                {'from': 'gpt', 'value': item["caption"]}
            ]
        ]
        # t2i_data_dict = preprocess_t2i(sources, self.tokenizer, vtokens_shape=0, p_drop_cond=0.)
        # prompt_token_ids = t2i_data_dict["input_ids"][0][0].tolist()
        data_dict = dict(
            prompt="<|t2i|>" + "A photo of " + item["caption"] + "<|soi|>",
            # prompt="<|t2i|>" + item["caption"] + "<|soi|>",
            # prompt_token_ids=prompt_token_ids,
            # image_path=item["image_path"]
        )
        return data_dict