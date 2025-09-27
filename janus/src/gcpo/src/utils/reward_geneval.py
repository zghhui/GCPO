import os
import os
import torch
import requests
from PIL import Image

import io
import pickle
import glob
import tqdm
from concurrent.futures import ThreadPoolExecutor
import os

class Geneval:
    def __init__(self, args):
        self.url = "http://127.0.0.1:18085"

    @property
    def __name__(self):
        return 'Geneval'
    
    def __call__(self, prompts, images, good_image, num_generations, geneval_meta_data=None, **kwargs):
        # image_list is a list of PIL image

        results = []
        batch_size = 2
        # print(geneval_meta_data)
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_meta_data = geneval_meta_data[i:i+batch_size] if isinstance(geneval_meta_data, list) else geneval_meta_data

            jpeg_data = []
            for image in batch_images:
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                jpeg_data.append(buffer.getvalue())
            # print(jpeg_data)
            data = {
                "images": jpeg_data,
                "meta_datas": batch_meta_data,
                "only_strict": False,
            }
            data_bytes = pickle.dumps(data)
            response = requests.post(self.url, data=data_bytes)
            response_data = pickle.loads(response.content)
            print(response_data)
            results.extend(response_data['rewards'])

        return results

"""
scores': [0.75, 1.0], 'rewards': [0.0, 1.0], 'strict_rewards': [0.0, 1.0], 'group_rewards': {'single_object': [-10.0, 1.0], 'two_object': [-10.0, -10.0], 'counting': [-10.0, -10.0], 'colors': [-10.0, -10.0], 'position': [-10.0, -10.0], 'color_attr': [0.0, -10.0]}, 'group_strict_rewards': {'single_object': [-10.0, 1.0], 'two_object': [-10.0, -10.0], 'counting': [-10.0, -10.0], 'colors': [-10.0, -10.0], 'position': [-10.0, -10.0], 'color_attr': [0.0, -10.0]}}
"""