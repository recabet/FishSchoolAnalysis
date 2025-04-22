import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights

import pandas as pd
from PIL import Image
import os
import time

from tqdm import tqdm

print("Loading ResNet-50 model...")

resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet50.eval()
feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

folder_path = '../data/image'
features_list = []
image_names = []

start_time = time.time()
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"Processing {len(image_files)} images...\n")


for filename in tqdm(image_files):
    try:
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            features = feature_extractor(img_tensor)
            features = features.view(-1).numpy()
        
        features_list.append(features)
        image_names.append(filename)
    
    except Exception as e:
        print(f"Skipping {filename}: {e}")

df = pd.DataFrame(features_list)
df.insert(0, 'image', image_names)


output_csv = '../data/resnet50_features.csv'
df.to_csv(output_csv, index=False)

end_time = time.time()

print(f"\n✅ Feature extraction complete!")
print(f"Saved {len(df)} image features to: {output_csv}")
print(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")
print(df.head())

