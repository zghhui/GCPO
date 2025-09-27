from pathlib import Path
import sys
sys.path.append("/home/bingxing2/ailab/quyichang/zgh/code/SimpleAR_LLamaGen_v2_no_cfg_aps")
import torch
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from PIL import Image
import os

# 指定文件夹路径
folder_path = '/home/bingxing2/ailab/quyichang/zgh/code/SimpleAR_LLamaGen_v2_no_cfg_aps/save/generated_images/step_200'

# 列出文件夹中的所有文件
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
image_files = []

# 遍历文件夹
for filename in os.listdir(folder_path):
    if any(filename.endswith(ext) for ext in image_extensions):
        image_files.append(os.path.join(folder_path, filename))
        
# SAMPLE_IMAGE_PATH = Path("/home/bingxing2/ailab/quyichang/zgh/code/SimpleAR_LLamaGen_v2_no_cfg_aps/save/generated_images/step_200/bs_0_rank_0_A_photo_of_a_woman_with_a_pencil_in_her_mouth.png")
asp_model_ckpt = '/home/bingxing2/ailab/quyichang/zgh/data/pretrain_model/models--google--siglip-so400m-patch14-384'
# load model and preprocessor
model, preprocessor =  convert_v2_5_from_siglip(
        predictor_name_or_path='/home/bingxing2/ailab/quyichang/zgh/data/pretrain_model/aesthetic_predictor_v2_5.pth',
        encoder_model_name=asp_model_ckpt,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
model = model.to(torch.bfloat16).cuda()

for image in image_files:
    image_name = image
    SAMPLE_IMAGE_PATH = Path(image)
    # load image to evaluate
    image = Image.open(SAMPLE_IMAGE_PATH).convert("RGB")

    # preprocess image
    pixel_values = (
        preprocessor(images=image, return_tensors="pt")
        .pixel_values.to(torch.bfloat16)
        .cuda()
    )

    # predict aesthetic score
    with torch.inference_mode():
        score = model(pixel_values).logits.squeeze().float().cpu().numpy()

    # print result
    print(image_name)
    print(f"Aesthetics score: {score:.2f}")