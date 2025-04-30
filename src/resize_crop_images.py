import os
import pandas as pd
import ast
from PIL import Image, ImageOps

# Load the DataFrame
df = pd.read_csv("../data/school_table_thresh.csv")

# Parse 'bbox' strings into tuples
df['bbox'] = df['bbox'].apply(ast.literal_eval)

# Output folder
output_dir = "../data/black_resized_croped_images"
os.makedirs(output_dir, exist_ok=True)

# Find the largest bbox size
max_width = 0
max_height = 0

for _, row in df.iterrows():
    bbox = row['bbox']
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    max_width = max(max_width, bbox_width)
    max_height = max(max_height, bbox_height)

# Process each image
for idx, row in df.iterrows():
    image_name = row['file_name']
    bbox = row['bbox']

    # Full path to the image
    image_path = os.path.join("../data/image", image_name + "_original.png")

    # Open the image
    image = Image.open(image_path)

    # Crop using bbox
    cropped_image = image.crop(bbox)

    # Calculate padding
    padding_left = (max_width - (bbox[2] - bbox[0])) // 2
    padding_top = (max_height - (bbox[3] - bbox[1])) // 2
    padding_right = max_width - (bbox[2] - bbox[0]) - padding_left
    padding_bottom = max_height - (bbox[3] - bbox[1]) - padding_top

    # Add padding
    padded_image = ImageOps.expand(
        cropped_image,
        (padding_left, padding_top, padding_right, padding_bottom),
        fill=(0, 0, 0)  # Black background
    )

    # Resize to desired size (width=86, height=246)
    resized_image = padded_image.resize((86, 246))

    # Save the result
    cropped_filename = os.path.join(output_dir, f"crop_{idx}.png")
    resized_image.save(cropped_filename)
