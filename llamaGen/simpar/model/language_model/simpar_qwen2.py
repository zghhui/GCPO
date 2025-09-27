#    Copyright 2024 Hao Zhang
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

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F

from transformers.generation.utils import GenerateOutput
from transformers import LogitsWarper, LogitsProcessorList, TemperatureLogitsWarper
from .modeling_qwen2 import Qwen2ForCausalLM


class CFGLogits(LogitsWarper):

    def __init__(self, cfg, unconditional_inputs, model, verbose=True):
        self.cfg = cfg
        self.inputs = unconditional_inputs
        self.model = model
        self.out = None
        self.verbose = verbose

    def __call__(self, input_ids, scores):
        scores = F.log_softmax(scores, dim=-1)
        if self.cfg == 1:
            return scores
        if self.out is None:
            self.out = self.model(self.inputs, use_cache=True)
        else:
            self.out = self.model(input_ids[:, -1:],
                                  use_cache=True,
                                  past_key_values=self.out.past_key_values)
        unconditional_logits = F.log_softmax(self.out.logits[:, -1], dim=-1)
        out = self.cfg * (scores - unconditional_logits) + unconditional_logits
        return out


class SimpARForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        Qwen2ForCausalLM.__init__(self, config)
    
    @torch.no_grad()
    def generate_visual_sjd(
        self,
        inputs,
        tokenizer,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        
        max_steps = kwargs["max_new_tokens"]
        height, width = int(math.sqrt(max_steps)), int(math.sqrt(max_steps))
        negative_prompt_ids = kwargs.pop("negative_prompt_ids", None)
        cfg_scale = kwargs.pop("cfg_scale", 1.0)
        temperature = kwargs.pop("temperature", 1.0)
        minumum_acc_tokens = kwargs.pop("minumum_acc_tokens", 1)

        device = inputs.device
        input_length = inputs.size(1)
        
        accepted_tokens = inputs.clone()
        generated_tokens = torch.tensor([], dtype=torch.long, device=device)
        draft_tokens = None
        prev_draft_probs = None
        
        for it in range(max_steps):
            # 1. Initialize Draft Tokens -------------------------------------------
            if draft_tokens is None:
                draft_tokens = torch.full(
                    (1, max_steps),
                    tokenizer.pad_token_id,
                    device=device
                )
            
            # 2. Forward Pass + CFG -----------------------------------------------
            cond_sequence = torch.cat([accepted_tokens, draft_tokens], dim=1)
            logits = self(cond_sequence).logits
            cond_draft_logits = logits[:, accepted_tokens.size(1)-1: -1, :]

            if cfg_scale > 1.0 and negative_prompt_ids is not None:
                neg_sequence = torch.cat([negative_prompt_ids, draft_tokens], dim=1)
                uncond_logits = self(neg_sequence).logits
                uncond_draft_logits = uncond_logits[:, negative_prompt_ids.size(1)-1: -1, :]
                draft_logits = uncond_draft_logits + cfg_scale * (cond_draft_logits - uncond_draft_logits)
            else:
                draft_logits = cond_draft_logits

            # 3. Temperature Scaling ----------------------------------------------
            draft_logits /= temperature
            draft_probs = F.softmax(draft_logits, dim=-1)

            # 4. First Iteration Handling -----------------------------------------
            if prev_draft_probs is None:
                # Greedy sample first token
                draft_tokens = torch.argmax(draft_probs, dim=-1)
                first_token = draft_tokens[:, :1]

                accepted_tokens = torch.cat([accepted_tokens, first_token], dim=1)
                generated_tokens = torch.cat([generated_tokens, first_token], dim=1)

                # Update negative prompt if provided
                if negative_prompt_ids is not None:
                    negative_prompt_ids = torch.cat([negative_prompt_ids, first_token], dim=1)
                
                prev_draft_probs = draft_probs[:, 1:].detach()
                draft_tokens = draft_tokens[:, 1:]
                continue  # Skip first iteration verification

            # 5. Speculative Verification -----------------------------------------
            # Get probabilities for current/previous draft tokens
            current_probs = draft_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
            prev_probs = prev_draft_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
            
            # Compute acceptance probability (Eq. 1)
            alpha = (current_probs / (prev_probs + 1e-8)).clamp(max=1.0)
            accept_mask = torch.rand_like(alpha) < alpha

            # at least accept several tokens to make sure acceleration
            accept_mask[:, 0: minumum_acc_tokens] = True

            # 6. Find First Rejection ---------------------------------------------
            rejected = (~accept_mask).nonzero(as_tuple=True)
            first_reject_pos = rejected[1][0].item() if len(rejected[0]) > 0 else draft_tokens.size(1)

            # 7. Update Accepted Tokens -------------------------------------------
            new_accepts = draft_tokens[:, :first_reject_pos]
            accepted_tokens = torch.cat([accepted_tokens, new_accepts], dim=1)
            # Update generated tokens
            generated_tokens = torch.cat([generated_tokens, new_accepts], dim=1)

            # Update negative prompt if provided
            if negative_prompt_ids is not None:
                negative_prompt_ids = torch.cat([negative_prompt_ids, new_accepts], dim=1)

            # 8. Early Termination Check ------------------------------------------
            if accepted_tokens.size(1) - input_length >= max_steps:
                print("Early termination at iter", it)
                break

            # 9. Resample Rejected Tokens (Eq. 2) ---------------------------------
            assert first_reject_pos < draft_tokens.size(1)
            # Compute calibrated probabilities for first rejected token
            residual_probs = (draft_probs - prev_draft_probs).clamp(min=0)
            residual_sum = residual_probs.sum(dim=-1, keepdim=True)  # Sum over token dimesion
            calibrated_probs = residual_probs / (residual_sum + 1e-8)
            
            calibrated_probs_slice = calibrated_probs[0, first_reject_pos]
            sampled_token = torch.multinomial(calibrated_probs_slice, num_samples=1).unsqueeze(0)
            first_reject_token = sampled_token


            rest_probs = draft_probs[0, first_reject_pos + 1:]
            rest_tokens = torch.multinomial(rest_probs, num_samples=1).transpose(0, 1)

            # Rebuild draft tokens and pad to max_steps
            draft_tokens = torch.cat([first_reject_token, rest_tokens], dim=1)
            # Update prev_draft_probs to match new_draft length
            prev_draft_probs = draft_probs[:, first_reject_pos: , :].detach()

        # 10. Final Output --------------------------------------------------------
        output = accepted_tokens[:, :input_length + max_steps]
        if output.size(1) < input_length + max_steps:
            padding = torch.full(
                (1, input_length + max_steps - output.size(1)),
                tokenizer.pad_token_id,
                device=device
            )
            output = torch.cat([output, padding], dim=1)
        
        return output

    @torch.no_grad()
    def generate_visual(
        self,
        inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        negative_prompt_ids = kwargs.pop("negative_prompt_ids", None)
        cfg_scale = kwargs.pop("cfg_scale", 1.0)
        
        return super().generate(
            inputs=inputs,
            position_ids=position_ids,
            attention_mask=attention_mask,
            logits_processor=LogitsProcessorList([
                CFGLogits(cfg_scale, negative_prompt_ids, self),
            ]),
            **kwargs
        )

