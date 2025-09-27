import os
import os
import torch
import requests
from PIL import Image
from torchvision import transforms
class Dinov2:
    def __init__(self, args):
        self.ckpt_path = args.dinov2_ckpt_path

    @property
    def __name__(self):
        return 'Dinov2'
    
    def load_to_device(self, load_device):
        self.model = torch.hub.load(
            self.ckpt_path, 
            'dinov2_vitl14',
            source='local'
        )
        self.model = self.model.to(load_device)
        self.model.eval()
        self.device = load_device
        
        # Construct image tranforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((378, 378)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    def calculate_distance(self, query_feature, database_features):
        query_feature = query_feature  # Add batch dimension to query_feature
        cosine_distances = [
            torch.dot(query_feature, feature) / (torch.norm(query_feature) * torch.norm(feature))
            for feature in database_features
        ]
        return cosine_distances
    def __call__(self, prompts, images, good_image, num_generations, **kwargs):
        # image_list is a list of PIL image
        
        ## read good image
        good_images = []
        for img_path_list in good_image:
            image_list = []
            for img_path in img_path_list:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = self.transform(img).unsqueeze(0)
                    image_list.append(img)
                    # print(img.shape)
                except Exception as e:
                    print(f"Warning: Image at {img_path} not found or failed to load. Error: {e}")
                    continue
            image_list = torch.cat(image_list).to(self.device)
            good_images.append(image_list)

        gen_images = []
        for i, (prompt, image) in enumerate(zip(prompts, images)):

            with torch.no_grad():
                # Process the image
                image = self.transform(image).unsqueeze(0)
                gen_images.append(image)
        gen_images = torch.cat(gen_images).to(self.device)
        gen_dino_features = self.model(gen_images)
        
        good_dino_features = []
        for good_image in good_images:
            good_dino_feature = self.model(good_image)
            good_dino_features.append(good_dino_feature)
        
        rewards = []
        for i in range(0, len(gen_dino_features)):
            gen_dino_feature = gen_dino_features[i]
            good_feature = good_dino_features[i // num_generations]
            cosine_distance = self.calculate_distance(gen_dino_feature, good_feature)
            mean_distance = torch.mean(torch.tensor(cosine_distance))  # Calculate the mean of the distances
            mean_distance = torch.clip(mean_distance, 0, 1)
            rewards.append(mean_distance.item())
            # rewards.append(1-mean_distance.item())
        return rewards

if __name__=='__main__':
    torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    