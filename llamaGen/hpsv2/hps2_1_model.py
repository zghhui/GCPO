import torch
from PIL import Image
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import warnings
import argparse
import os
import requests
from clint.textui import progress
from typing import Union
import huggingface_hub
from hpsv2.utils import root_path, hps_version_map

warnings.filterwarnings("ignore", category=UserWarning)

def create_hps_model(model_name = "ViT-H-14", precision = 'amp', checkpoint_path='./', device='cuda'):
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name,
        None,
        precision=precision,
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device)
    model.eval()
    print('Loading model successfully!')
    return model, tokenizer, preprocess_val