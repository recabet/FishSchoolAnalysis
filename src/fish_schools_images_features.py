'''Based on this images, this is a Python script that can extract relevant features from such images. 
The code will focus on:
Identifying distinct regions (like the blue/purple and yellow areas)
Calculating the size/area of these regions
Measuring density based on color intensity
Analyzing depth distribution
Extracting texture features that could help with classification.
This code extracts multiple features from fish school images:
Color-based segmentation - Divides the image into distinct regions (like the blue and yellow areas)
Size measurements - Calculates area and perimeter of each segment
Density analysis - Measures intensity variations within each identified region
Depth distribution - Analyzes how fish schools are distributed vertically
Texture features - Extracts texture patterns that can help identify different school types
Edge detection - Identifies boundaries between different regions.
The output will include visualizations and a CSV file with all extracted features
The code will:
Process all images in your folder
Create visualizations for each image
Generate a CSV file with all extracted features
Save everything to an 'output' folder 

In the fish school feature extraction code, we focused on several key features that are particularly useful 
for classifying fish schools:

Color-based segmentation features:
Percentage of image occupied by each identified segment
Mean color values of each segment

Size and shape measurements:
Area of each detected fish school region
Perimeter of regions
Compactness ratio (perimeter²/area) which helps distinguish between compact and dispersed schools

Density analysis:
Mean intensity within each segment (indicates density of fish)
Standard deviation of intensity (shows how uniform the school is)
Edge density (helps identify school boundaries)

Depth distribution:
Mean depth of each segment
Minimum and maximum depths where schools appear
Depth range of each school
Vertical distribution profile

Texture features (using Gray Level Co-occurrence Matrix):
Contrast (measures local variations)
Homogeneity (measures similarity of pixels)
Energy (measures uniformity)
Correlation (measures how correlated a pixel is to its neighbors)
Dissimilarity (measures how different pixels are)

These features are effective for echogram/sonar images, where color intensity, spatial distribution, 
and texture patterns can help distinguish between different types of fish schools, 
their behaviors, and potentially even species.

These features are important for fish school classification for several reasons:
Color-based features capture fundamental differences in acoustic reflection properties. 
In sonar/echogram imagery, different colors represent varying echo intensities, which correlate with fish 
density, size, or species. 
Size and shape measurements help distinguish between different schooling behaviors. 
Some species form tight, compact schools while others form looser aggregations. 
The compactness ratio specifically helps identify whether schools are dense and 
circular or more irregular and dispersed.
Density analysis is critical because fish school density is often species-specific. 
Some fish naturally school very tightly (like herring or sardines) while others maintain greater 
distances between individuals. Density variation within a school can also indicate predator presence or 
feeding behavior.
Depth distribution features are important as different species occupy different parts of the water column. 
Some prefer surface waters while others stay in mid-water or near the bottom. 
Depth range also captures vertical migration behaviors that can be species-specific.
Texture features reveal internal structure patterns within schools that aren't 
captured by simple density measures. These subtle patterns can help distinguish between species 
with similar overall densities but different internal organization.

Together, these features create a multi-dimensional fingerprint of each fish school that machine 
learning algorithms can use to classify them by species, behavior, or other characteristics of interest.

'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops
from skimage import measure

def load_image(image_path):
    """Load an image and convert to RGB if needed."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def extract_features(image):
    """Extract features from fish school images."""
    features = {}
    
    # Convert to grayscale for some processing
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 1. Color-based segmentation using K-means
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3  # Assuming we're looking for 3 main regions (can be adjusted)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to 8 bit values
    centers = np.uint8(centers)
    
    # Flatten the labels array
    labels = labels.flatten()
    
    # Create segmented image
    segmented_image = centers[labels].reshape(image.shape)
    
    # Count pixels in each segment
    unique_labels, counts = np.unique(labels, return_counts=True)
    segment_percentages = {f"segment_{i}_percentage": count/len(labels)*100 for i, count in zip(unique_labels, counts)}
    features.update(segment_percentages)
    
    # 2. Size features
    # Create binary masks for each segment
    masks = {}
    for i in unique_labels:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[labels.reshape(gray.shape) == i] = 255
        masks[i] = mask
        
        # Calculate area and perimeter for each segment
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            area = sum(cv2.contourArea(contour) for contour in contours)
            perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
            features[f"segment_{i}_area"] = area
            features[f"segment_{i}_perimeter"] = perimeter
            if area > 0:
                features[f"segment_{i}_compactness"] = (perimeter**2) / area
    
    # 3. Density features - mean intensity within each segment
    for i in unique_labels:
        mask = masks[i]
        mean_intensity = cv2.mean(gray, mask=mask)[0]
        features[f"segment_{i}_mean_intensity"] = mean_intensity
        
        # Standard deviation of intensity (density variation)
        mask_binary = mask > 0
        if np.any(mask_binary):
            segment_pixels = gray[mask_binary]
            features[f"segment_{i}_intensity_std"] = np.std(segment_pixels)
    
    # 4. Depth distribution
    # Assuming depth increases from top to bottom of image
    height, width = gray.shape
    depth_profiles = {}
    
    for i in unique_labels:
        mask = masks[i] > 0
        if np.any(mask):
            # Create a profile of presence across depth
            depth_profile = np.sum(mask, axis=1) / width  # Normalize by width
            depth_profiles[i] = depth_profile
            
            # Calculate depth metrics
            indices = np.arange(height)
            if np.sum(depth_profile) > 0:
                mean_depth = np.sum(indices * depth_profile) / np.sum(depth_profile)
                features[f"segment_{i}_mean_depth"] = mean_depth
                
                # Find min and max depths where segment is present
                present_depths = indices[depth_profile > 0]
                if len(present_depths) > 0:
                    features[f"segment_{i}_min_depth"] = present_depths.min()
                    features[f"segment_{i}_max_depth"] = present_depths.max()
                    features[f"segment_{i}_depth_range"] = present_depths.max() - present_depths.min()
    
    # 5. Texture features using GLCM (Gray Level Co-occurrence Matrix)
    distances = [1, 3, 5]  # px
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # radians
    
    for i in unique_labels:
        mask = masks[i]
        if np.sum(mask > 0) > 100:  # Only if we have enough pixels
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
            
            # Need to crop to non-zero region for GLCM
            coords = cv2.findNonZero(mask)
            if coords is not None and len(coords) > 100:
                x, y, w, h = cv2.boundingRect(coords)
                roi = masked_gray[y:y+h, x:x+w]
                
                # Replace zeros (from masking) with the mean value to avoid artifacts
                if np.any(roi > 0):
                    mean_val = np.mean(roi[roi > 0])
                    roi[roi == 0] = mean_val
                    
                    # Calculate GLCM properties
                    glcm = graycomatrix(roi, distances, angles, 256, symmetric=True, normed=True)
                    
                    # Calculate properties
                    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
                    for prop in props:
                        feature = graycoprops(glcm, prop)
                        features[f"segment_{i}_{prop}"] = feature.mean()
    
    # 6. Edge density (can help identify school boundaries)
    edges = cv2.Canny(gray, 50, 150)
    features['overall_edge_density'] = np.sum(edges > 0) / (height * width)
    
    # For each segment, calculate edge density
    for i in unique_labels:
        mask = masks[i]
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
        if np.sum(mask > 0) > 0:
            features[f"segment_{i}_edge_density"] = np.sum(masked_edges > 0) / np.sum(mask > 0)
    
    return features, segmented_image, masks

def analyze_directory(directory_path):
    """Process all images in a directory and extract features."""
    results = {}
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            image_path = os.path.join(directory_path, filename)
            try:
                image = load_image(image_path)
                features, _, _ = extract_features(image)
                results[filename] = features
                print(f"Processed {filename}, extracted {len(features)} features")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return results

def visualize_features(image_path, save_dir=None):
    """Generate visualizations of the feature extraction process."""
    image = load_image(image_path)
    features, segmented_image, masks = extract_features(image)
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axs[0, 0].imshow(image)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    
    # Segmented image
    axs[0, 1].imshow(segmented_image)
    axs[0, 1].set_title('Segmented Image')
    axs[0, 1].axis('off')
    
    # Edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    axs[0, 2].imshow(edges, cmap='gray')
    axs[0, 2].set_title('Edge Detection')
    axs[0, 2].axis('off')
    
    # Mask visualization for up to 3 segments
    for i, (segment_id, mask) in enumerate(masks.items()):
        if i < 3:
            axs[1, i].imshow(mask, cmap='gray')
            axs[1, i].set_title(f'Segment {segment_id} Mask')
            axs[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.basename(image_path).split('.')[0]
        plt.savefig(os.path.join(save_dir, f"{base_name}_features.png"))
    else:
        plt.show()
    
    # Print key features
    print(f"Key features for {os.path.basename(image_path)}:")
    for k, v in features.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")
    
    return features

def export_features_to_csv(results, output_path):
    """Export extracted features to a CSV file."""
    import pandas as pd
    
    # Convert to DataFrame
    all_features = set()
    for features in results.values():
        all_features.update(features.keys())
    
    # Create DataFrame with all possible features
    df = pd.DataFrame(index=results.keys(), columns=list(all_features))
    
    # Fill in the values
    for image_name, features in results.items():
        for feature, value in features.items():
            df.loc[image_name, feature] = value
    
    # Fill NaN values with 0
    df = df.fillna(0)
    
    # Save to CSV
    df.to_csv(output_path)
    print(f"Features exported to {output_path}")
    
    return df

def main():
    """Main function demonstrating usage."""
    # Example usage
    # Replace 'path/to/images' with your directory containing fish school images
    image_directory = 'C://UFAZ//Elgun_Rajab_Omar_Ismail//R_O_I//image'
    output_directory = 'output'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Process a single image for visualization
    example_image = os.path.join(image_directory, os.listdir(image_directory)[0])
    visualize_features(example_image, save_dir=output_directory)
    
    # Process all images
    results = analyze_directory(image_directory)
    
    # Export features
    export_features_to_csv(results, os.path.join(output_directory, 'fish_school_features.csv'))
    
    print(f"Feature extraction complete. Results saved to {output_directory}")

if __name__ == "__main__":
    main()