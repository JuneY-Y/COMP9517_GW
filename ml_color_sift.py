import cv2
import numpy as np
from PIL import Image

# Function to extract color SIFT features from an image.
def extract_color_sift_features(image, n_features=100, color_space='rgb'):

    # Check if image is grayscale
    if len(image.shape) < 3:
        # For grayscale images, use regular SIFT
        return extract_standard_sift(image, n_features)
    
    # Convert to RGB if needed (OpenCV uses BGR by default)
    if color_space != 'bgr':
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
    
    # Convert to the requested color space
    if color_space == 'rgb' or color_space == 'rgbnorm':
        # Ensure RGB order (OpenCV uses BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_color = image
            
        # For normalized RGB, convert to RGB norm
        if color_space == 'rgbnorm':
            image_color = normalize_rgb(image_color)
            
    elif color_space == 'opponent':
        # Convert to opponent color space
        image_color = rgb_to_opponent(image)
    elif color_space == 'hsv':
        # Convert to HSV
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError(f"Unsupported color space: {color_space}")
    
    # Split into channels
    channels = cv2.split(image_color)
    
    # Process each channel
    all_descriptors = []
    
    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=n_features)
    
    for i, channel in enumerate(channels):
        # Ensure channel is in uint8 format
        if channel.dtype != np.uint8:
            channel = (channel * 255).astype(np.uint8)
        
        # Skip completely uniform channels (like HSV channels for grayscale images)
        if np.min(channel) == np.max(channel):
            continue
            
        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(channel, None)
        
        # If no keypoints were found, skip this channel
        if descriptors is None:
            continue
        
        # Add channel-specific descriptors to the list
        all_descriptors.append(descriptors)
    
    # Combine descriptors from all channels
    if not all_descriptors:
        # If no descriptors were found at all, return empty array with correct shape
        return np.zeros((0, 128))


    combined_descriptors = np.vstack(all_descriptors)
    
    # Ensure we have at least some features
    if combined_descriptors.shape[0] < 5:
        # Generate some random descriptors to avoid empty feature vectors
        random_descriptors = np.random.randn(5, 128).astype(np.float32)
        # Normalize them to have similar scale to SIFT descriptors
        for i in range(random_descriptors.shape[0]):
            random_descriptors[i] /= np.linalg.norm(random_descriptors[i])
        return random_descriptors
    
    return combined_descriptors

# Function to extract standard SIFT features from grayscale image
def extract_standard_sift(image, n_features=100):

    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=n_features)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # If no keypoints were found, return empty array with correct shape
    if descriptors is None:
        # SIFT descriptors are 128-dimensional
        return np.zeros((0, 128)) 
    
    # Ensure we have at least some sift features
    if descriptors.shape[0] < 5:
        # Generate some random descriptors to avoid empty feature vectors
        random_descriptors = np.random.randn(5, 128).astype(np.float32)
        # Normalize them to have similar scale to SIFT descriptors
        for i in range(random_descriptors.shape[0]):
            random_descriptors[i] /= np.linalg.norm(random_descriptors[i])
        return random_descriptors
    
    return descriptors

# Function to convert RGB image to opponent color space
def rgb_to_opponent(image):

    # Ensure the image is in RGB (not BGR)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Split into R, G, B channels
    R, G, B = cv2.split(image_rgb)
    R = R.astype(np.float32)
    G = G.astype(np.float32)
    B = B.astype(np.float32)
    
    # Create opponent color channels
    # O1 = (R-G)/sqrt(2)
    O1 = (R - G) / np.sqrt(2)
    # O2 = (R+G-2B)/sqrt(6)
    O2 = (R + G - 2*B) / np.sqrt(6)
    # O3 = (R+G+B)/sqrt(3)
    O3 = (R + G + B) / np.sqrt(3)
    
    # Normalize to 0-255 range for SIFT
    O1 = cv2.normalize(O1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    O2 = cv2.normalize(O2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    O3 = cv2.normalize(O3, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Merge channels
    opponent = cv2.merge([O1, O2, O3])
    return opponent

# Function to convert RGB to normalized RGB color space
def normalize_rgb(image):

    # Split into R, G, B channels
    R, G, B = cv2.split(image)
    R = R.astype(np.float32)
    G = G.astype(np.float32)
    B = B.astype(np.float32)
    
    # Calculate sum
    sum_rgb = R + G + B
    # Avoid division by zero
    sum_rgb[sum_rgb == 0] = 1
    
    # Normalize each channel
    r_norm = (R / sum_rgb * 255).astype(np.uint8)
    g_norm = (G / sum_rgb * 255).astype(np.uint8)
    b_norm = (B / sum_rgb * 255).astype(np.uint8)
    
    # Merge channels
    rgb_norm = cv2.merge([r_norm, g_norm, b_norm])
    return rgb_norm

# Extract SIFT features from multiple color spaces and combine them.
# This approach is more robust for aerial imagery where different
# color spaces can capture different types of features
def extract_multi_color_sift(image, n_features=100):

    # Define color spaces to use
    color_spaces = ['rgb', 'opponent', 'hsv']
    
    # Extract features from each color space
    all_descriptors = []
    for color_space in color_spaces:
        descriptors = extract_color_sift_features(image, n_features, color_space)
        if descriptors.shape[0] > 0:
            all_descriptors.append(descriptors)
    
    # Combine all descriptors
    if not all_descriptors:
        return np.zeros((0, 128))
    
    combined_descriptors = np.vstack(all_descriptors)
    return combined_descriptors

# Function to create a bag of visual words
def create_bag_of_visual_words(descriptors_list, n_clusters=50):

    # Stack all descriptors
    if not descriptors_list:
        print("Warning: No descriptors found for codebook creation. Creating default codebook.")
        return np.zeros((n_clusters, 128))
    
    # Filter out empty descriptor arrays
    valid_descriptors = [d for d in descriptors_list if d.shape[0] > 0]
    
    if not valid_descriptors:
        print("Warning: No valid descriptors found. Creating default codebook.")
        return np.zeros((n_clusters, 128))
    
    all_descriptors = np.vstack(valid_descriptors)
    
    if all_descriptors.shape[0] < n_clusters:
        print(f"Warning: Number of descriptors ({all_descriptors.shape[0]}) is less than requested clusters ({n_clusters}).")
        print(f"Reducing number of clusters to {all_descriptors.shape[0]}.")
        n_clusters = all_descriptors.shape[0]
    
    print(f"Creating codebook with {n_clusters} clusters from {all_descriptors.shape[0]} descriptors.")
    
    # Create and train KMeans
    try:
        kmeans = cv2.kmeans(
            all_descriptors.astype(np.float32),
            n_clusters,
            None,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
            attempts=10,
            flags=cv2.KMEANS_RANDOM_CENTERS
        )
        
        return kmeans[2]  # Return the cluster centers
    
    except Exception as e:
        print(f"Error in K-means clustering: {e}")
        print("Creating fallback random codebook.")
        return np.random.randn(n_clusters, 128).astype(np.float32)

def compute_bovw_features(descriptors, codebook):

    # Initialize histogram
    hist = np.zeros(len(codebook))
    
    # Exit if no descriptors were found
    if descriptors.shape[0] == 0:
        return hist
    
    # normalise the descriptors and codebook for better distance calculation
    normalised_descriptors = np.copy(descriptors)
    for i in range(normalised_descriptors.shape[0]):
        norm = np.linalg.norm(normalised_descriptors[i])
        if norm > 0:
            normalised_descriptors[i] /= norm
    
    normalised_codebook = np.copy(codebook)
    for i in range(normalised_codebook.shape[0]):
        norm = np.linalg.norm(normalised_codebook[i])
        if norm > 0:
            normalised_codebook[i] /= norm
    
    # Assign each descriptor to the nearest visual word using Euclidean distance
    for descriptor in normalised_descriptors:
        distances = np.sqrt(((normalised_codebook - descriptor)**2).sum(axis=1))
        nearest_word = np.argmin(distances)
        hist[nearest_word] += 1
    
    # normalise histogram
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    
    return hist