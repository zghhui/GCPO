import yaml
import re
import os
import math
import json
import time
import copy
import random
import numpy as np
from typing import Dict

import torch
from torch.utils.data import Dataset
import transformers
from simpar.train.preprocess import preprocess_multimodal, preprocess_t2i, preprocess_t2v
from simpar.utils import rank0_print

class T2TokenDataset(Dataset):
    def __init__(self, data_path: str, data_dir: str, tokenizer: transformers.PreTrainedTokenizer, data_args=None, p_drop_cond=0.0, sample_short=False):
        super(T2TokenDataset, self).__init__()
        
        self.data_path = data_path
        self.data_dir = data_dir
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

            rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
            self.list_data_dict.extend(cur_data_dict)
        
        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.p_drop_cond = p_drop_cond
        self.sample_short = sample_short

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # label_path = sample["label_path"]
            # labels = open(label_path, "r").readlines()
            cur_len = 1 # len(labels[0].split())
            # cur_len = len(sample["caption"].split())
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            # if "image" in sample or "video" in sample or self.data_args.early_mix_text:
            length_list.append(cur_len)
            
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3

        # try the current sample first
        try:
            sample = self._get_item(i)
            if sample is not None:
                return sample
        except Exception as e:
            # sleep 1s in case it is a cloud disk issue
            print(f"Failed to fetch sample {i}. Exception:", e)

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

        
        sample = self._get_item(0)
        return sample

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        token_path = sources["code_path"] if self.data_dir is None else os.path.join(self.data_dir, sources["code_path"])
        label_path = sources["label_path"] if self.data_dir is None else os.path.join(self.data_dir, sources["label_path"])
        data_type = "image"
        
        with open(label_path) as f:
            prompts = f.readlines()
        
        if len(prompts) > 1:
            if self.sample_short:
                prompt = random.choice(prompts).strip()
            else:
                prompt = prompts[1].strip()
        elif len(prompts) == 0:
            prompt = ""
            rank0_print("Empty prompt")
        else:
            prompt = prompts[0].strip()
            # find first '.'
            idx = prompt.find('.')
            if idx != -1:
                short_prompt = prompt[:idx + 1]
            else:
                short_prompt = prompt
            
            prompts = [short_prompt, prompt]
            if self.sample_short:
                prompt = random.choice(prompts).strip()
            else:
                prompt = prompts[1].strip()

        codes = np.load(token_path)
        codes = codes.reshape(codes.shape[0], -1)
        
        vtokens_shape = codes.shape[1]
        codes = torch.LongTensor(codes).squeeze(0) + len(self.tokenizer)
        sources = [
            {
                "conversations": [{'from': 'human', 'value': '<image>'}, {'from': 'gpt', 'value': prompt}]
            }
        ]
        
        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), data_args=self.data_args)
        t2i_data_dict = preprocess_t2i(sources, self.tokenizer, vtokens_shape=vtokens_shape, p_drop_cond=self.p_drop_cond)
        
        input_ids = t2i_data_dict["input_ids"][0][0]
        labels = t2i_data_dict["labels"][0][0]

        vtoken_placehold_id = self.tokenizer.convert_tokens_to_ids(["<|vtokens|>"])
        vtoken_placehold_id = torch.tensor(vtoken_placehold_id)
        place_holder_mask = (input_ids == vtoken_placehold_id)
        place_holder_mask_tgt = (labels == vtoken_placehold_id)
        
        input_ids[place_holder_mask] = codes
        labels[place_holder_mask_tgt] = codes
            
        data_dict = dict(
            input_ids=t2i_data_dict["input_ids"][0][0],
            labels=t2i_data_dict["labels"][0][0],
            data_type="image_gen",
            codes=codes
        )
        return data_dict
