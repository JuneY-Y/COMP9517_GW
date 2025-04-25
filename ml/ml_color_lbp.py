import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern

def extract_color_lbp_features(image, n_points=24, radius=8, method='uniform'):

    # Check if image is grayscale or color
    if len(image.shape) < 3:
        # For grayscale images, use regular LBP
        gray = image
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # Compute LBP
        lbp = local_binary_pattern(gray, n_points, radius, method)
        # Compute histogram of LBP
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return hist
    
    # For color images, compute LBP on each channel
    # Split the image into its color channels
    channels = cv2.split(image)
    all_hists = []
    
    # Process each channel
    for channel in channels:
        # Apply Gaussian blur to reduce noise
        channel_blurred = cv2.GaussianBlur(channel, (3, 3), 0)
        
        # Compute LBP for this channel
        lbp = local_binary_pattern(channel_blurred, n_points, radius, method)
        
        # Compute histogram of LBP
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        # Add to list of histograms
        all_hists.append(hist)
    
    # Concatenate histograms from all channels
    combined_hist = np.concatenate(all_hists)
    
    # Normalize the combined histogram
    if np.sum(combined_hist) > 0:
        combined_hist = combined_hist / np.sum(combined_hist)
    
    return combined_hist

# Advanced version that also extracts opponent color LBP features
def extract_advanced_color_lbp_features(image, n_points=24, radius=8, method='uniform', use_opponent=True):

    if len(image.shape) < 3:
        # For grayscale images, revert to standard LBP
        return extract_color_lbp_features(image, n_points, radius, method)
    
    all_hists = []
    
    # Process RGB channels
    channels = cv2.split(image)
    for channel in channels:
        # Apply Gaussian blur to reduce noise
        channel_blurred = cv2.GaussianBlur(channel, (3, 3), 0)
        
        # Compute LBP for this channel
        lbp = local_binary_pattern(channel_blurred, n_points, radius, method)
        
        # Compute histogram of LBP
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        # Add to list of histograms
        all_hists.append(hist)
    
    # If enabled, process opponent color space (more discriminative for aerial imagery)
    if use_opponent and len(channels) == 3:
        # Create opponent color channels: O1 = (R-G)/sqrt(2), O2 = (R+G-2B)/sqrt(6), O3 = (R+G+B)/sqrt(3)
        R, G, B = channels
        R = R.astype(np.float32)
        G = G.astype(np.float32)
        B = B.astype(np.float32)
        
        O1 = (R - G) / np.sqrt(2)
        O2 = (R + G - 2 * B) / np.sqrt(6)
        O3 = (R + G + B) / np.sqrt(3)
        
        # Normalize to 0-255 range for LBP
        O1 = cv2.normalize(O1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        O2 = cv2.normalize(O2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        O3 = cv2.normalize(O3, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Process each opponent channel
        for opponent_channel in [O1, O2, O3]:
            # Apply Gaussian blur
            channel_blurred = cv2.GaussianBlur(opponent_channel, (3, 3), 0)
            
            # Compute LBP
            lbp = local_binary_pattern(channel_blurred, n_points, radius, method)
            
            # Compute histogram
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            
            # Add to list of histograms
            all_hists.append(hist)
    
    # Concatenate all histograms
    combined_hist = np.concatenate(all_hists)
    
    # Normalize the combined histogram
    if np.sum(combined_hist) > 0:
        combined_hist = combined_hist / np.sum(combined_hist)
    
    return combined_hist

# Function to extract multi-scale Color LBP features
def extract_multiscale_color_lbp(image, radii=[1, 3, 5, 8], n_points=24, method='uniform', use_opponent=False):

    # Initialize list to store histograms for each scale
    multiscale_hists = []
    
    # Extract LBP features at each scale (radius)
    for radius in radii:
        if use_opponent:
            hist = extract_advanced_color_lbp_features(image, n_points, radius, method, use_opponent)
        else:
            hist = extract_color_lbp_features(image, n_points, radius, method)
        multiscale_hists.append(hist)
    
    # Concatenate histograms from all scales
    combined_hist = np.concatenate(multiscale_hists)
    
    # Normalize the combined histogram
    if np.sum(combined_hist) > 0:
        combined_hist = combined_hist / np.sum(combined_hist)
    
    return combined_hist
