# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
# from transformers import Qwen2VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from open_r1.trainer import JanusGCPOTrainer
from torch.utils.data import Dataset
import json
import random
import numpy as np
import torch

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# add image_generation_prompt in the GRPOConfig
@dataclass
class GRPOConfig(GRPOConfig):
    """
    Configuration class for the GRPO training script.
    """
    new_generations_image: int = field(default=1, metadata={"help": "The number of new generations of image to generate"})
    image_token_num_per_image: int = field(default=576, metadata={"help": "The number of image tokens to generate"})
    # image_gen_temperature: float = field(default=1.0, metadata={"help": "The temperature for image generation"}) # HACK, this is always 1.0
    cfg_weight: float = field(default=3.0, metadata={"help": "The cfg weight for image generation"})
    reasoning_prompt_path: Optional[str] = field(
        default='',
    )
    img_size: int = field(default=384, metadata={"help": "The size of the image to generate"})
    patch_size: int = field(default=16, metadata={"help": "The patch size of the image to generate"})
    max_textcot_length: int = field(default=None, metadata={"help": "The maximum length of the text cot"})
    hps_ckpt_path: str = field(default=None, metadata={"help": "The path to the hps checkpoint"})
    git_ckpt_path: str = field(default=None, metadata={"help": "The path to the git checkpoint"})
    gdino_ckpt_path: str = field(default=None, metadata={"help": "The path to the gdino checkpoint"})
    gdino_config_path: str = field(default=None, metadata={"help": "The path to the gdino config"})
    orm_ckpt_path: str = field(default=None, metadata={"help": "The path to the orm checkpoint"})
    dinov2_ckpt_path: str = field(default=None, metadata={"help": "The path to the dinov2 checkpoint"})
    clip_ckpt_path: str = field(default=None, metadata={"help": "The path to the clip checkpoint"})
    pickscore_ckpt_path: str = field(default=None, metadata={"help": "The path to the pickscore checkpoint"})
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'hps', 'git', 'gdino'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["hps", "git", "gdino", "orm", "dinov2", "pickscore", "geneval"],
        metadata={"help": "List of reward functions. Possible values: 'hps', 'git', 'gdino', 'orm', 'dinov2, 'pickscore','geneval'"}
    )
    img_save_dir: str = field(default=None, metadata={"help": "The path to the img save"})
    data_dir: str = field(default=None, metadata={"help": "The path to the img dir"})

def make_detection_prompt(nouns):
    if len(nouns) == 0:
        return '', []
    
    token_spans = []
    pointer = 0
    for noun in nouns:
        n_split = noun.strip().split(" ")
        if len(n_split) == 1:
            length = len(n_split[0])
            token_spans.append([[pointer, pointer + length]])
            pointer += length + 3 # on the blank space after the noun
        else: # multiple words
            beg_len = len(n_split[0])
            total_length = len(noun)
            end_len = len(n_split[-1])
            token_spans.append([[pointer, pointer + beg_len], [pointer + total_length - end_len, pointer + total_length]])
            pointer += total_length + 3 # on the blank space after the noun
    text_prompt = ' . '.join(nouns) + "." # need to end with '.
    return text_prompt, token_spans


reward_funcs_registry = {
    "hps": 'hps',
    'hps_compare': 'hps_compare',
    'git': 'git',
    'gdino': 'gdino',
    'orm': 'orm',
    'unify': 'unify',
    'dinov2': 'dinov2',
    'pickscore': 'pickscore',
    'geneval': 'geneval'
}

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = dataset
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {
            "prompt": [
                {"role": "User", "content": self.prompts[idx]},
                {"role": "Assistant", "content": ""},
            ],
            "raw_prompt": self.prompts[idx],
            "meta_datas": self.metadatas[idx]
        }

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

# Format into conversation
def make_conversation(example):
    return {
        "prompt": [
            {"role": "User", "content": example["prompt"]},
            {"role": "Assistant", "content": ""},
        ],
        "raw_prompt": example["prompt"],
    }
                
def main(script_args, training_args, model_args):

    trainer_cls = JanusGCPOTrainer
    print("using: ", trainer_cls)
    
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    
    print(script_args.reward_funcs)
    if script_args.reward_funcs == ["geneval"]:
        dataset = GenevalPromptDataset(script_args.dataset_name, 'train')

        trainer = trainer_cls(
            model=model_args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset if training_args.eval_strategy != "no" else None,
            peft_config=get_peft_config(model_args),
            attn_implementation=model_args.attn_implementation,
            script_args=script_args,
        )
    
    else:       
        # Load the dataset
        if script_args.dataset_name.endswith('.csv'):
            suffix = 'csv'
        elif script_args.dataset_name.endswith('.json'):
            suffix = 'json'
        elif script_args.dataset_name.endswith('.parquet'):
            suffix = 'parquet'
        dataset = load_dataset(suffix, data_files=script_args.dataset_name)
        # print('Dataset length: ', len(dataset['train']))
        dataset = dataset.map(make_conversation)

        # Initialize the GRPO trainer
        trainer = trainer_cls(
            model=model_args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            peft_config=get_peft_config(model_args),
            attn_implementation=model_args.attn_implementation,
            script_args=script_args,
        )

    trainer.img_save_dir = script_args.img_save_dir
    trainer.geneval_style = script_args.reward_funcs == ["geneval"]

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    seed_all(42)
    main(script_args, training_args, model_args)
