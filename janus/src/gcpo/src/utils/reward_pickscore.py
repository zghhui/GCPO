import os
import os
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel

class PickScore:
    def __init__(self, args):
        self.pickscore_ckpt_path = args.pickscore_ckpt_path
        self.clip_ckpt_path = args.clip_ckpt_path

    @property
    def __name__(self):
        return 'PickScore'
    
    def load_to_device(self, load_device):
        self.processor = AutoProcessor.from_pretrained(self.clip_ckpt_path)
        self.model = AutoModel.from_pretrained(self.pickscore_ckpt_path).eval().to(load_device)
        self.device = load_device
    
    def __call__(self, prompts, images, good_image, num_generations, **kwargs):
        # image_list is a list of PIL image
        device = self.device
        result = []
        
        
        for i, (prompt, image) in enumerate(zip(prompts, images)):
            # preprocess
            image_inputs = self.processor(
                images=image,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            pixel_values = image_inputs["pixel_values"]
            text_inputs = self.processor(
                text=[prompt],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ) 
            input_ids = text_inputs["input_ids"]
            attention_mask = text_inputs["attention_mask"]
            # embed
            # print(image_inputs.shape)
            # print(text_inputs.shape)
            # exit()
            with torch.no_grad():
                image_embs = self.model.get_image_features(pixel_values=pixel_values)
                image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            
                text_embs = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
                text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
            
                # score
                scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
                
                result.append(scores.item()/100)
            # get probabilities if you have multiple images to choose from
            # probs = torch.softmax(scores, dim=-1)
            
        return result