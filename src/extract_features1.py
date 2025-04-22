import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel
from scipy.stats import entropy, skew, kurtosis
from tqdm import tqdm



def compute_edge_density(img_gray):
    #Calculate edge density using Sobel filter
    edges = sobel(img_gray)
    return np.mean(edges)

def extract_features(image_path:str):
    #Extract features from a single image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Basic features
    mean = np.mean(img)
    std = np.std(img)
    
    # Entropy Build histogram of pixel values (0–255). 
    # Counts how many pixels fall into each gray level.
    # Converts it to probabilities: divides by total number of pixels.
    # Calculates Shannon entropy from those probabilities.
    # so we got the global entropy, computed on the full image 
    hist = np.histogram(img, bins=256, range=(0, 255))[0]
    hist_prob = hist / hist.sum()
    ent = entropy(hist_prob, base=2)
    
    # GLCM Contrast
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    
    # Statistical features
    skewness = skew(img.flatten())
    kurt = kurtosis(img.flatten())
    
    # Edge density (new feature)
    edge_density = compute_edge_density(img)
    
    return {
        'mean': mean,
        'std': std,
        'entropy': ent,
        'contrast': contrast,
        'skewness': skewness,
        'kurtosis': kurt,
        'edge_density': edge_density  # Added feature
    }

def batch_process_images(image_dir, output_csv="features.csv"):
    #Process all images and save features to CSV
    image_paths = [os.path.join(image_dir, f) 
                  for f in os.listdir(image_dir) 
                  if f.endswith('.png')]
    
    features_list = []
    for path in tqdm(image_paths, desc="Processing Images"):
        features = extract_features(path)
        if features:
            features['image_id'] = os.path.basename(path)
            features_list.append(features)
    
    df = pd.DataFrame(features_list)
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

# Run feature extraction with edge density
# batch_process_images(r"C:\\UFAZ\\Elgun_Rajab_Omar_Ismail\\R_O_I\\image")

