import os
import cv2
import pandas as pd

img_dir = "../data/image"
output_csv = "../data/flattened_image.csv"

# Initialize CSV with header for the first image
first_image = True

for image_name in os.listdir(img_dir):
    image_path = os.path.join(img_dir, image_name)
    image_arr = cv2.imread(image_path)

    if image_arr is None:
        print(f"Warning: Could not load image {image_name}. Skipping.")
        continue

    # Resize to (1000x86) as (width, height)
    resized = cv2.resize(image_arr, (1000, 86))

    # Flatten to 1D vector
    flat = resized.flatten()

    # Create a DataFrame for a single image
    df = pd.DataFrame([flat])

    # Write the header only once
    df.to_csv(output_csv, mode='w' if first_image else 'a', index=False, header=first_image)
    first_image = False

print("All images processed and written to CSV incrementally.")
