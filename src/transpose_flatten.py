import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

img_dir = "../data/black_resized_croped_images"
data = []
df = pd.DataFrame()

for image_name in tqdm(os.listdir(img_dir), desc="Processing images"):
    image_path = os.path.join(img_dir, image_name)
    image_arr = cv2.imread(image_path)

    # Convert to black and white (grayscale)
    gray_image = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)

    # Flatten to 1D vector
    flat = gray_image.astype(np.float32).flatten()

    data.append(flat)

df = pd.DataFrame(data)
df.to_csv("../data/black_flattened_image.csv", index=False)
