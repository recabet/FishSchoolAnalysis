import os
import cv2
import pandas as pd

img_dir = "../data/image"
data = []

for image_name in os.listdir(img_dir):
    image_path = os.path.join(img_dir, image_name)
    image_arr = cv2.imread(image_path)

    # Resize to (1000x86) as (width, height)
    resized = cv2.resize(image_arr, (1000, 86))

    # Flatten to 1D vector
    flat = resized.flatten()

    data.append(flat)

# Create DataFrame: each row = 1 image
df = pd.DataFrame(data)
df.to_csv("../data/flattened_image.csv")

print(f"Loaded {df.shape[0]} images with {df.shape[1]} features each.")
