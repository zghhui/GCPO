#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch

from transformers import AutoTokenizer, BitsAndBytesConfig

from simpar.model import *
from simpar.utils import rank0_print
from simpar.model.language_model.simpar_qwen2 import SimpARForCausalLM

def load_pretrained_model(model_path, device_map="auto", attn_implementation="flash_attention_2", **kwargs):
    kwargs["device_map"] = device_map
    kwargs["torch_dtype"] = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = SimpARForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, **kwargs)

    rank0_print(f"Model Class: {model.__class__.__name__}")
    image_processor = None
    context_len = 2048

    return tokenizer, model, image_processor, context_len


def vllm_t2i(model_path, device_map="bfloat16"):
    from vllm import LLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LLM(model=model_path, tokenizer=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.2, use_v2_block_manager=True, dtype=device_map)
    return tokenizer, model