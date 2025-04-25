import numpy as np
from PIL import Image

# import color LBP feature extraction function
from ml_color_lbp import extract_multiscale_color_lbp

# import color SIFT feature extraction functions
from ml_color_sift import extract_multi_color_sift, create_bag_of_visual_words, compute_bovw_features

# Class to handle feature extraction from images
class FeatureExtractor:
    
    def __init__(self, feature_type='combined', use_codebook=True):
        self.feature_type = feature_type
        self.use_codebook = use_codebook
        self.codebook = None
        self.fixed_lbp_size = None
        self.use_opponent = True
        
        # SIFT parameters
        self.sift_n_features = 100
        self.sift_color_space = 'opponent' 

        # LBP parameters
        self.lbp_n_points = 24
        self.lbp_method = 'uniform'
        self.lbp_radii = [1, 3, 5, 8]
        self.lbp_radius = 8
    
    def extract_from_tensor(self, tensor):
        # Convert tensor to numpy (H, W, C)
        image = tensor.permute(1, 2, 0).cpu().numpy()
        
        # Convert from normalised range back to [0, 255]
        # First check if we need denormalisation (values in range approximately [-1, 1])
        if image.min() < 0 or image.max() <= 1.0:
            # Standard ImageNet normalisation was applied
            # Undo normalisation: first multiply by std then add back mean
            image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            # Scale from [0, 1] to [0, 255]
            image = (image * 255).astype(np.uint8)
        
        return self.extract_from_numpy(image)
    
    def extract_from_numpy(self, image):
        features = []
        
        if self.feature_type == 'sift' or self.feature_type == 'combined':
            if self.use_codebook and self.codebook is not None:
                # Extract SIFT features and compute bag of visual words
                descriptors = extract_multi_color_sift(image, self.sift_n_features)
                bovw_features = compute_bovw_features(descriptors, self.codebook)
                features.append(bovw_features)
            else:
                # Just extract SIFT descriptors for codebook creation
                descriptors = extract_multi_color_sift(image, self.sift_n_features)
                if descriptors.shape[0] > 0:
                    return descriptors
        
        if self.feature_type == 'lbp' or self.feature_type == 'combined':
            # Extract LBP features
            lbp_features = extract_multiscale_color_lbp(image, self.lbp_radii, self.lbp_n_points, self.lbp_method, self.use_opponent)
            # Store standard size if not already set
            if self.fixed_lbp_size is None:
                self.fixed_lbp_size = len(lbp_features)
            # Ensure consistent size by padding or truncating
            elif len(lbp_features) != self.fixed_lbp_size:
                if len(lbp_features) < self.fixed_lbp_size:
                    # Pad with zeros
                    lbp_features = np.pad(lbp_features, (0, self.fixed_lbp_size - len(lbp_features)))
                else:
                    # Truncate
                    lbp_features = lbp_features[:self.fixed_lbp_size]
                    
            features.append(lbp_features)
        
        # Combine features
        if len(features) > 0:
            if len(features) == 1:
                return features[0]
            else:
                return np.concatenate(features)
        
        return None
    
    def create_codebook(self, dataloader, n_clusters=100):
        print("Extracting SIFT descriptors for codebook creation...")
        all_descriptors = []
        
        # Extract SIFT descriptors from all images in the dataloader
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx % 5 == 0:
                print(f"  Processing batch {batch_idx}/{len(dataloader)}")
            
            # Process each image in the batch
            for img in images:
                # Convert to numpy and extract features
                img_np = img.permute(1, 2, 0).cpu().numpy()
                
                # Undo normalisation properly
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = (img_np * 255).astype(np.uint8)
                
                descriptors = extract_multi_color_sift(img_np, n_features=200) # Extract more features
                if descriptors.shape[0] > 0:
                    all_descriptors.append(descriptors)
        
        # Create the codebook
        # Increase n_clusters for better representation
        self.codebook = create_bag_of_visual_words(all_descriptors, n_clusters=max(n_clusters, 200))
        return self.codebook
    
    def extract_features_batch(self, dataloader):
        features = []
        labels = []
        
        # Set up progress tracking
        total_batches = len(dataloader)
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx % 5 == 0:
                print(f"  Processing batch {batch_idx}/{total_batches} ({batch_idx/total_batches*100:.1f}%)")
            
            batch_features = []
            
            for img in images:
                img_features = self.extract_from_tensor(img)
                
                # Handle the case where features might be None
                if img_features is None:
                    # Use a default feature vector with appropriate dimensionality
                    if self.feature_type == 'sift':
                        default_size = len(self.codebook) if self.codebook is not None else 100
                    elif self.feature_type == 'lbp':
                        default_size = self.fixed_lbp_size or 26  # Default LBP size
                    else:  # combined
                        codebook_size = len(self.codebook) if self.codebook is not None else 100
                        lbp_size = self.fixed_lbp_size or 26
                        default_size = codebook_size + lbp_size
                        
                    img_features = np.zeros(default_size)
                
                batch_features.append(img_features)
            
            features.extend(batch_features)
            labels.extend(targets.cpu().numpy())
        
        # Convert to numpy arrays and ensure consistent shapes
        features_array = np.array(features)
        labels_array = np.array(labels)
        
        # Check for NaN or infinity values
        if np.isnan(features_array).any() or np.isinf(features_array).any():
            print("Warning: NaN or infinity values detected in features. Replacing with zeros.")
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features_array, labels_array
