import os
import pandas as pd
import ast
from PIL import Image, ImageOps
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy, skew, kurtosis
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df = pd.read_csv("C://UFAZ//Elgun_Rajab_Omar_Ismail//R_O_I//school_table_thresh.csv")
df['bbox'] = df['bbox'].apply(ast.literal_eval)

image_dir = "C://UFAZ//Elgun_Rajab_Omar_Ismail//R_O_I//image"
output_dir = "../data/cropped_images"
os.makedirs(output_dir, exist_ok=True)

# Find maximum bbox size to standardize padded output
max_width = max(row['bbox'][2] - row['bbox'][0] for _, row in df.iterrows())
max_height = max(row['bbox'][3] - row['bbox'][1] for _, row in df.iterrows())

features = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    image_name = row['file_name'] + "_original.png"
    bbox = row['bbox']
    image_path = os.path.join(image_dir, image_name)

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_name}: {e}")
        continue

    # Crop the image using bbox
    cropped = image.crop(bbox)

    # Pad image to max dimensions For each bounding box in the CSV, 
    # the corresponding region (fish school) is cropped from the image.
    # It’s padded so that all cropped images have the same dimensions, 
    # for model input consistency.
    padding = [
        (max_width - cropped.width) // 2,
        (max_height - cropped.height) // 2
    ]
    padded = ImageOps.expand(cropped, (padding[0], padding[1],
                                       max_width - cropped.width - padding[0],
                                       max_height - cropped.height - padding[1]), fill=(255, 255, 255))

    # Convert to grayscale and numpy array for texture features. 
    # Converts the cropped image to grayscale for texture/statistical analysis.
    # Converts to HSV for color-based analysis 
    # it is important as the fish schools are in blue, and seabed/surface in yellow.
    gray = padded.convert('L')
    img_gray = np.array(gray)

    # Texture features, Gray features: Mean, standard deviation, entropy for texture, 
    # contrast GLCM Haralick features quantify texture using Gray-Level Co-occurrence Matrices, 
    # skewness, kurtosis.
    # HSV features: Average Hue, Saturation, and Value
    # useful for identifying water blue vs seabed/surface yellow.
    # These features give us a fingerprint of each fish school zone
    mean_val = np.mean(img_gray)
    std_val = np.std(img_gray)
    hist = np.histogram(img_gray, bins=256, range=(0, 255))[0]
    hist_prob = hist / hist.sum()
    ent = entropy(hist_prob, base=2)
    contrast = graycoprops(graycomatrix(img_gray, [1], [0], 256, True, True), 'contrast')[0, 0]
    skew_val = skew(img_gray.flatten())
    kurt_val = kurtosis(img_gray.flatten())

    # Color features in HSV
    hsv = np.array(padded.convert('HSV'))
    h_mean, s_mean, v_mean = hsv[..., 0].mean(), hsv[..., 1].mean(), hsv[..., 2].mean()
    h_std, s_std, v_std = hsv[..., 0].std(), hsv[..., 1].std(), hsv[..., 2].std()

    features.append({
        'image_id': row['file_name'],
        'mean': mean_val,
        'std': std_val,
        'entropy': ent,
        'contrast': contrast,
        'skewness': skew_val,
        'kurtosis': kurt_val,
        'hue_mean': h_mean,
        'saturation_mean': s_mean,
        'value_mean': v_mean,
        'hue_std': h_std,
        'saturation_std': s_std,
        'value_std': v_std,
        'depth': row['depth'],
        'bbox_width': bbox[2] - bbox[0],
        'bbox_height': bbox[3] - bbox[1]
    })

# Convert to DataFrame
feat_df = pd.DataFrame(features)

# Normalize features, use StandardScaler to ensure all features contribute equally to clustering 
#  as entropy may vary differently ...compared to other features like hue.
scaler = StandardScaler()
numeric_cols = [col for col in feat_df.columns if col not in ['image_id']]
feat_df_scaled = pd.DataFrame(scaler.fit_transform(feat_df[numeric_cols]), columns=numeric_cols)
feat_df_scaled['image_id'] = feat_df['image_id']

# Save to CSV
feat_df_scaled.to_csv("fish_school_features.csv", index=False)
print("Feature extraction and normalization complete.")

# Optional: Visualize clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(feat_df_scaled[numeric_cols])


kmeans = KMeans(n_clusters=4, random_state=0).fit(X_pca)
feat_df_scaled['cluster'] = kmeans.labels_

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans.labels_, palette="viridis")
plt.title("KMeans Clustering of Fish School Features (PCA-Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()