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
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
import os
import sys
import logging
import wandb
from typing import Union, Any, Optional
from dataclasses import dataclass, field
from PIL import Image

# get dir_root simpar/train/llamaGen_trainer_gcpo.py, 2
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import transformers
import datasets
from transformers import set_seed, PreTrainedModel, AutoTokenizer, PreTrainedTokenizerBase, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from accelerate.utils import broadcast_object_list, gather, gather_object, set_seed
import open_clip

from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer, load_openai_model
from hpsv2.hps2_1_model import create_hps_model
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.trainer.gcpo_llama_trainer import GRPOTrainer
from trl.trainer.utils import pad
from trl.models import unwrap_model_for_generation
from simpar.model.llamagen.vq_model import VQ_16
from simpar.model.tokenizer.cosmos_tokenizer.networks import TokenizerConfigs
from simpar.model.tokenizer.cosmos_tokenizer.video_lib import CausalVideoTokenizer as CosmosTokenizer
from simpar.train.t2i_data import GRPOT2IDataset
from simpar.grpo.configs import GRPOConfig
from simpar.grpo.utils.callbacks import get_callbacks
from simpar.grpo.utils.wandb_logging import init_wandb_training
from simpar.grpo.utils.swanlab_logging import init_swanlab_training
from simpar.grpo.rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
    clip_reward,
    aesthetic_reward,
    hps_reward,
    aps_reward
)
from transformers import CLIPProcessor, CLIPModel, T5Tokenizer
import numpy as np
logger = logging.getLogger(__name__)

from simpar.model.llama_model import LlamaForCausalLM
from typing import Any, Callable, Optional, Sized, Union
from datasets import Dataset, IterableDataset
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from collections import defaultdict
from torchvision.utils import save_image
import re
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy

def clean_prompt_string(prompt):
    # Remove characters that are not alphanumeric, underscore, or space
    cleaned_prompt = re.sub(r'[^a-zA-Z0-9_\- ]', '', prompt)
    return cleaned_prompt.replace(" ", "_")

class LlamaGenGRPOTrainer(GRPOTrainer):

    def _decode_images(self, completion_ids, prompts_text=None):
        device = self.accelerator.device
        rank = self.accelerator.process_index
        with torch.inference_mode():
            embeddings, generated_images = self.model.decode_ids(completion_ids, self.vq_model)
        
        if self.state.global_step % self.save_image_steps == 0:
            save_dir = os.path.join(self.args.output_dir, f"generated_images/step_{self.state.global_step}")
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存原始分辨率图像
            for i, img in enumerate(generated_images):
                if prompts_text is not None:
                    prompt_clean_i = prompts_text[i].strip("<|t2i|>").strip("<|soi|>")
                    prompt_clean_i = clean_prompt_string(prompt_clean_i)  # Clean the prompt
                    save_image(img, f'{save_dir}/bs_{i}_rank_{rank}_{prompt_clean_i}.png', normalize=True, value_range=(-1, 1))
                
                else:
                    save_image(img, f'{save_dir}/bs_{i}.png', normalize=True, value_range=(-1, 1))          
        # resize to 224 to save memory
        generated_images = torch.nn.functional.interpolate(generated_images, size=(224, 224), mode="bilinear", align_corners=False)
        generated_images = (255 * (generated_images * 0.5 + 0.5)).clamp(0, 255)
        PIL_image = []
        for i, img_tensor in enumerate(generated_images):
            # 将每个图像转换为 numpy 数组
            img_array = img_tensor.permute(1, 2, 0).byte().cpu().numpy()
            
            # 创建一个 PIL.Image 对象
            img_pil = Image.fromarray(img_array)
            PIL_image.append(img_pil)
        return PIL_image, embeddings

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [p for p in prompts]   
        generate_params = {
            "prompt": prompts_text,
            "max_length": self.latent_size ** 2,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "cfg": 1.0,
        }   

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                ordered_set_of_prompts = list(dict.fromkeys(all_prompts_text))
                if len(ordered_set_of_prompts) < 7:
                    ordered_set_of_prompts = ordered_set_of_prompts + ordered_set_of_prompts[:7 - len(ordered_set_of_prompts)]

                all_outputs = self.llm.generate(
                    ordered_set_of_prompts, sampling_params=self.sampling_params, use_tqdm=False
                )
                completion_ids = []
                for outputs in all_outputs:
                    for output in outputs.outputs:
                        completion_ids.append(output.token_ids)
                
                decoded_images, decoded_image_embeds = self._decode_images(completion_ids) # List of images [C, H, W]
                
            else:
                completion_ids = [None] * len(all_prompts_text)
                decoded_images = [None] * len(all_prompts_text)
                decoded_image_embeds = [None] * len(all_prompts_text)

            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]
            decoded_images = broadcast_object_list(decoded_images, from_process=0)
            decoded_images = decoded_images[process_slice]

            decoded_image_embeds = broadcast_object_list(decoded_image_embeds, from_process=0)
            decoded_image_embeds = decoded_image_embeds[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                # prompt_completion_ids = unwrapped_model.generate(
                #     prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                # )
                prompt_ids, prompt_mask, completion_ids, completion_ids_prob_list = unwrapped_model.generate(**generate_params)
                # completion_ids_prob_list  = [completion_ids_prob.unsqueeze(0) for completion_ids_prob in completion_ids_prob_list]
                completion_ids_prob = torch.stack(completion_ids_prob_list, dim=1) # [b, s, vocab_size]
                entropy = entropy_from_logits(completion_ids_prob) # [b, s]
                seq_len = entropy.size(1)
                 
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                decoded_images, embeddings = self._decode_images(completion_ids, prompts_text) # List of images [C, H, W]
        
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        with torch.inference_mode():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )      
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        completions = []
        for i, (prompt, image) in enumerate(zip(prompts, decoded_images)):
            prompt_clean = prompt.strip("<|t2i|>").strip("<|soi|>")
            with torch.inference_mode():
                
                ## HPS Reward
                hps_text = self.hps_tokenizer([prompt_clean.strip()]).to(device=device, non_blocking=True)  
                hps_image = self.hps_preprocess_val(image).unsqueeze(0).to(device=device, non_blocking=True)                   
                with torch.amp.autocast('cuda'):
                    hps_outputs = self.hps_model(hps_image, hps_text)
                    hps_image_features, hps_text_features = hps_outputs["image_features"], hps_outputs["text_features"]
            
            completions.append(
                [{
                    "hps_image_features": hps_image_features,
                    "hps_text_features": hps_text_features
                }]
            )
        del decoded_images, prompt_completion_ids, attention_mask, logits_to_keep
        torch.cuda.empty_cache()
        
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):  
            output_reward_func = reward_func(prompts=prompts, completions=completions)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        
        entropy_clone = entropy.clone()
        # 记录不同区间的熵均值到 metrics
        if self.metric_entropy:
            entropy = gather(entropy)
            seq_len = entropy.size(1)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "embeddings": embeddings,
            "entropy": entropy_clone
        }
    

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )

    data_path: str = field(
        default="",
        metadata={"help": "Path to the generated data"},
    )

    vq_model_ckpt: str = field(
        default="/path_to_tokenizer/Cosmos-1.0-Tokenizer-DV8x16x16"
    )
    clip_model_ckpt: str = field(
        default="/path_to_clip/vit_large_patch14_clip_224.openai"
    )
    aest_model_ckpt: str = field(
        default="/path_to_aesthetic/aesthetic-predictor/sa_0_4_vit_l_14_linear.pth"
    )
    asp_model_ckpt: str = field(
        default="/path_to_aesthetic/aesthetic-predictor/sa_0_4_vit_l_14_linear.pth"
    )
    hps_model_path: str = field(
        default="/path_to_aesthetic/aesthetic-predictor/sa_0_4_vit_l_14_linear.pth"
    )
    text_tokenizer: str = field(
        default="/path_to_text_tokenizer/flan-t5-xl"
    )
    image_size: int = field(
        default=512,
        metadata={"help": "Image size"},
    )
    downscale_factor: int = field(
        default=16,
        metadata={"help": "Downscale factor"},
    )
    dataset_train_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use for training"},
    )
    ## 新参数
    metric_entropy: bool = field(
        default=True,
        metadata={"help": "Whether to compute and log entropy metrics during training"},
    )
    cfg: float = field(
        default=7.5,
        metadata={"help": "CFG"},
    )
    save_image_steps: int = field(
        default=25,
        metadata={"help": "Steps to save generated images"},
    )

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)
    elif "swanlab" in training_args.report_to:
        init_swanlab_training(training_args)

    ################
    # Load tokenizer
    ################
    # print(training_args)
    # tokenizer = T5Tokenizer.from_pretrained(script_args.text_tokenizer, use_fast=False, device_map='auto')
    tokenizer = T5Tokenizer.from_pretrained(script_args.text_tokenizer, use_fast=False)

    # Load VQ model
    vq_model = VQ_16(codebook_size=16384, codebook_embed_dim=8)
    checkpoint = torch.load(script_args.vq_model_ckpt)
    vq_model.load_state_dict(checkpoint['model'])
    vq_model = vq_model.to("cuda")
    vq_model.eval()
    
    hps_model, hps_tokenizer, hps_preprocess_val = create_hps_model(checkpoint_path=script_args.hps_model_path)
    
    # Load the dataset
    dataset = GRPOT2IDataset(data_path=script_args.data_path, tokenizer=tokenizer)

    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": code_reward,
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
        "clip": clip_reward,
        "aesthetic": aesthetic_reward,
        "hps": hps_reward,
        "aps": aps_reward
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # Initialize the GRPO trainer
    #############################
    ## You can use wandb or swanlab callback for logging
    from swanlab.integration.transformers import SwanLabCallback
    swanlab_callback = SwanLabCallback(training_args.wandb_project)
    trainer = LlamaGenGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args)+[swanlab_callback],
        processing_class=tokenizer,
    )
    trainer.vq_model = vq_model
    trainer.latent_size = script_args.image_size // script_args.downscale_factor

    trainer.hps_model = hps_model
    trainer.hps_tokenizer = hps_tokenizer
    trainer.hps_preprocess_val = hps_preprocess_val
    
    ## new
    trainer.metric_entropy = script_args.metric_entropy
    trainer.cfg = script_args.cfg
    trainer.save_image_steps = script_args.save_image_steps

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics
    # metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    # logger.info("*** Save model ***")
    # trainer.save_model(training_args.output_dir)
    # logger.info(f"Model saved to {training_args.output_dir}")

    # # Save everything else on main process
    # kwargs = {
    #     "dataset_name": script_args.dataset_name,
    #     "tags": ["open-r1"],
    # }
    # if trainer.accelerator.is_main_process:
    #     trainer.create_model_card(**kwargs)
    #     # Restore k,v cache for fast inference
    #     trainer.model.config.use_cache = True
    #     trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)