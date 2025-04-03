import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib

# ---------------------- From your previous code ----------------------
# Reuse the AerialImageDataGenerator class adapted for feature extraction

class AerialImageDataGenerator:
    def __init__(self, batch_size=32, img_height=224, img_width=224, augmentation=True):
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.augmentation = augmentation
        
        # Basic ImageDataGenerator with standard augmentations
        self.basic_datagen = ImageDataGenerator(
            rescale=1./255,  # Normalize pixel values
            validation_split=0.2  # 20% for validation
        )
        
        # Advanced ImageDataGenerator with more augmentations
        self.aug_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,  # Rotate up to 20 degrees
            width_shift_range=0.2,  # Horizontal shift
            height_shift_range=0.2,  # Vertical shift
            brightness_range=(0.4, 1.2),  # Custom brightness range from your existing code
            zoom_range=[1.0, 1.5],  # Custom zoom range from your existing code
            horizontal_flip=True,
            vertical_flip=True,  # Useful for aerial imagery
            fill_mode='nearest',
            validation_split=0.2  # 20% for validation
        )
    
    def load_and_prepare_data(self, folder_path):
        """
        Load images from folder structure and prepare generators
        Expected structure: folder_path/category/image.jpg
        """
        # Choose the appropriate data generator
        datagen = self.aug_datagen if self.augmentation else self.basic_datagen
        
        # Create train generator
        train_generator = datagen.flow_from_directory(
            folder_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Create validation generator
        validation_generator = datagen.flow_from_directory(
            folder_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, validation_generator

# ---------------------- Feature Extraction with SIFT and LBP ----------------------

def extract_sift_features(image, n_features=100):
    """
    Extract SIFT features from an image
    
    Args:
        image: Input image (RGB or grayscale)
        n_features: Maximum number of features to extract
        
    Returns:
        Array of SIFT descriptors
    """
    # Convert to grayscale if needed
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
        return np.zeros((0, 128))  # SIFT descriptors are 128-dimensional
    
    return descriptors

def extract_lbp_features(image, n_points=24, radius=8, method='uniform'):
    """
    Extract Local Binary Pattern features from an image
    
    Args:
        image: Input image (RGB or grayscale)
        n_points: Number of points to consider
        radius: Radius of circle
        method: LBP method (uniform, default, etc.)
        
    Returns:
        LBP histogram features
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method)
    
    # Compute histogram of LBP
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist

def create_bag_of_visual_words(descriptors_list, n_clusters=50):
    """
    Create a bag of visual words model using KMeans clustering
    
    Args:
        descriptors_list: List of SIFT descriptors from training images
        n_clusters: Number of visual words (clusters)
        
    Returns:
        Trained KMeans model
    """
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
    """
    Compute bag of visual words features for a set of descriptors
    
    Args:
        descriptors: SIFT descriptors from an image
        codebook: Visual words (cluster centers)
        
    Returns:
        Histogram of visual words
    """
    # Initialize histogram
    hist = np.zeros(len(codebook))
    
    # Exit if no descriptors were found
    if descriptors.shape[0] == 0:
        return hist
    
    # Assign each descriptor to the nearest visual word
    for descriptor in descriptors:
        distances = np.sqrt(((codebook - descriptor)**2).sum(axis=1))
        nearest_word = np.argmin(distances)
        hist[nearest_word] += 1
    
    # Normalize histogram
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    
    return hist

def extract_features_with_augmentation(folder_path, n_augmentations=3, feature_type='combined', sample_size=None, random_seed=42):
    """
    Extract features from original and augmented images
    
    Args:
        folder_path: Path to the dataset organized in folders (one per class)
        n_augmentations: Number of augmented versions to generate per image
        feature_type: 'sift', 'lbp', or 'combined'
        sample_size: Number of images to sample per class (None = use all)
        random_seed: Random seed for reproducibility
    
    Returns:
        X_train, y_train, X_val, y_val - Features and labels for training and validation
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Initialize data generator
    img_height, img_width = 224, 224
    batch_size = 32
    
    # Create data generator instance without augmentation for feature extraction
    data_gen = AerialImageDataGenerator(
        batch_size=batch_size,
        img_height=img_height,
        img_width=img_width,
        augmentation=False  # We'll manually control augmentation
    )
    
    # Sample images if sample_size is specified
    if sample_size is not None:
        print(f"Sampling {sample_size} images per class...")
        
        # Get list of all image files per class
        sampled_files = []
        
        for class_name in os.listdir(folder_path):
            class_path = os.path.join(folder_path, class_name)
            if os.path.isdir(class_path):
                # Get all image files for this class
                all_files = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                             if f.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp'))]
                
                # Sample randomly
                if len(all_files) > sample_size:
                    sampled = np.random.choice(all_files, size=sample_size, replace=False)
                else:
                    sampled = all_files
                
                sampled_files.extend(sampled)
        
        # Create temporary directory for sampled images
        import tempfile
        import shutil
        
        sample_dir = tempfile.mkdtemp()
        
        # Copy sampled files to temporary directory maintaining class structure
        for file_path in sampled_files:
            # Get class name (parent directory name)
            class_name = os.path.basename(os.path.dirname(file_path))
            
            # Create class directory in sample directory
            os.makedirs(os.path.join(sample_dir, class_name), exist_ok=True)
            
            # Copy file
            dest_path = os.path.join(sample_dir, class_name, os.path.basename(file_path))
            shutil.copy(file_path, dest_path)
        
        # Update folder path to use the sampled directory
        folder_path = sample_dir
    
    # Load and prepare data
    train_generator, validation_generator = data_gen.load_and_prepare_data(folder_path)
    
    # Create augmentation generator for training data
    augmentation_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=(0.4, 1.2),
        zoom_range=[1.0, 1.5],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    # Get class information
    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    
    # Process all batches from training generator
    print("Processing training images...")
    X_train_images = []
    y_train_labels = []
    sift_descriptor_list = []  # For creating codebook
    
    for i in range(len(train_generator)):
        X_batch, y_batch = train_generator[i]
        
        # Add original images and labels
        for j in range(len(X_batch)):
            X_train_images.append(X_batch[j])
            y_train_labels.append(np.argmax(y_batch[j]))
            
            # Extract SIFT descriptors for codebook (only from original images)
            img = (X_batch[j] * 255).astype(np.uint8)
            descriptors = extract_sift_features(img)
            if descriptors is not None and descriptors.shape[0] > 0:
                sift_descriptor_list.append(descriptors)
        
        # Generate augmented versions
        for _ in range(n_augmentations):
            # Create augmented batch
            aug_gen = augmentation_gen.flow(X_batch, y_batch, batch_size=batch_size, shuffle=False)
            X_aug, y_aug = next(aug_gen)
            
            # Add augmented images and labels
            for j in range(len(X_aug)):
                X_train_images.append(X_aug[j])
                y_train_labels.append(np.argmax(y_aug[j]))
    
    # Process all batches from validation generator
    print("Processing validation images...")
    X_val_images = []
    y_val_labels = []
    
    for i in range(len(validation_generator)):
        X_batch, y_batch = validation_generator[i]
        
        # Add original images and labels (no augmentation for validation)
        for j in range(len(X_batch)):
            X_val_images.append(X_batch[j])
            y_val_labels.append(np.argmax(y_batch[j]))
    
    # Create bag of visual words codebook
    print("Creating bag of visual words codebook...")
    codebook = create_bag_of_visual_words(sift_descriptor_list, n_clusters=100)
    
    # Extract features based on selected type
    print(f"Extracting {feature_type} features for training images...")
    X_train_features = []
    fixed_lbp_size = None  # Will store the standard size for LBP features
    
    for img in X_train_images:
        img = (img * 255).astype(np.uint8)
        
        if feature_type == 'sift' or feature_type == 'combined':
            # Extract SIFT features and compute bag of visual words
            descriptors = extract_sift_features(img)
            bovw_features = compute_bovw_features(descriptors, codebook)
        
        if feature_type == 'lbp' or feature_type == 'combined':
            # Extract LBP features
            lbp_features = extract_lbp_features(img)
            
            # Store standard size if not already set
            if fixed_lbp_size is None:
                fixed_lbp_size = len(lbp_features)
            # Ensure consistent size by padding or truncating
            elif len(lbp_features) != fixed_lbp_size:
                if len(lbp_features) < fixed_lbp_size:
                    # Pad with zeros
                    lbp_features = np.pad(lbp_features, (0, fixed_lbp_size - len(lbp_features)))
                else:
                    # Truncate
                    lbp_features = lbp_features[:fixed_lbp_size]
        
        # Combine features
        if feature_type == 'sift':
            features = bovw_features
        elif feature_type == 'lbp':
            features = lbp_features
        else:  # combined
            features = np.concatenate([bovw_features, lbp_features])
        
        X_train_features.append(features)
    
    print(f"Extracting {feature_type} features for validation images...")
    X_val_features = []
    
    for img in X_val_images:
        img = (img * 255).astype(np.uint8)
        
        if feature_type == 'sift' or feature_type == 'combined':
            # Extract SIFT features and compute bag of visual words
            descriptors = extract_sift_features(img)
            bovw_features = compute_bovw_features(descriptors, codebook)
        
        if feature_type == 'lbp' or feature_type == 'combined':
            # Extract LBP features
            lbp_features = extract_lbp_features(img)
            
            # Ensure consistent size by padding or truncating
            if len(lbp_features) != fixed_lbp_size:
                if len(lbp_features) < fixed_lbp_size:
                    # Pad with zeros
                    lbp_features = np.pad(lbp_features, (0, fixed_lbp_size - len(lbp_features)))
                else:
                    # Truncate
                    lbp_features = lbp_features[:fixed_lbp_size]
        
        # Combine features
        if feature_type == 'sift':
            features = bovw_features
        elif feature_type == 'lbp':
            features = lbp_features
        else:  # combined
            features = np.concatenate([bovw_features, lbp_features])
        
        X_val_features.append(features)
    
    # Convert to numpy arrays
    X_train = np.array(X_train_features)
    y_train = np.array(y_train_labels)
    X_val = np.array(X_val_features)
    y_val = np.array(y_val_labels)
    
    print(f"Feature extraction complete: Train {X_train.shape}, Validation {X_val.shape}")
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Save codebook for future use
    np.save('sift_codebook.npy', codebook)
    
    # Clean up temporary directory if we created one
    if sample_size is not None:
        try:
            import shutil
            shutil.rmtree(folder_path)
            print(f"Cleaned up temporary sample directory")
        except:
            print(f"Note: Failed to clean up temporary directory {folder_path}")
    
    return X_train, y_train, X_val, y_val, class_names

def train_ml_models(X_train, y_train, X_val, y_val, class_names):
    """
    Train SVM, KNN, and K-means models using the extracted features
    
    Args:
        X_train, y_train: Training features and labels
        X_val, y_val: Validation features and labels
        class_names: Dictionary mapping class indices to class names
    
    Returns:
        Dictionary of trained models
    """
    # Apply PCA to reduce dimensionality (optional but helps with high-dimensional features)
    pca_n_components = min(100, X_train.shape[1])  # Limit to 100 components or feature dimension
    
    # List of models to train
    models = {
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=pca_n_components)),
            ('svm', SVC(kernel='rbf', probability=True, C=10, gamma='scale'))
        ]),
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=pca_n_components)),
            ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
        ]),
        'K-means': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=pca_n_components)),
            ('kmeans', KMeansClassifier(n_clusters=len(class_names)))
        ])
    }
    
    # Train and evaluate each model
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, target_names=list(class_names.values()))
        conf_matrix = confusion_matrix(y_val, y_pred)
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"{name} Classification Report:\n{report}")
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {name}')
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, list(class_names.values()), rotation=45)
        plt.yticks(tick_marks, list(class_names.values()))
        
        # Add text annotations in the confusion matrix
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        
        # Save the model
        joblib.dump(model, f'{name.replace(" ", "_").lower()}_model.pkl')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'conf_matrix': conf_matrix
        }
    
    return results

# ---------------------- K-means Classifier Wrapper ----------------------

class KMeansClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that wraps KMeans clustering for classification tasks.
    This adapts K-means (unsupervised) to work as a classifier (supervised).
    """
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_to_label_map = {}
        
    def fit(self, X, y):
        # Fit K-means
        self.kmeans.fit(X)
        
        # Map each cluster to the most common class label
        clusters = self.kmeans.predict(X)
        for cluster in range(self.n_clusters):
            # Find indices of samples in this cluster
            indices = np.where(clusters == cluster)[0]
            if len(indices) > 0:
                # Get the most common class label in this cluster
                most_common_label = np.bincount(y[indices]).argmax()
                self.cluster_to_label_map[cluster] = most_common_label
            else:
                # Handle empty clusters
                self.cluster_to_label_map[cluster] = -1
                
        return self
    
    def predict(self, X):
        # Predict clusters
        clusters = self.kmeans.predict(X)
        
        # Map clusters to class labels
        return np.array([self.cluster_to_label_map.get(c, -1) for c in clusters])
    
    def predict_proba(self, X):
        # This is a rough approximation for probability
        # based on distance to cluster centers
        distances = self.kmeans.transform(X)
        
        # Convert distances to similarities (closer = higher probability)
        similarities = 1 / (1 + distances)
        
        # Normalize to get probabilities
        probabilities = similarities / similarities.sum(axis=1, keepdims=True)
        
        # Reorder to match class labels
        n_classes = max(self.cluster_to_label_map.values()) + 1
        result = np.zeros((X.shape[0], n_classes))
        
        for cluster, label in self.cluster_to_label_map.items():
            if label >= 0:  # Skip -1 labels (empty clusters)
                result[:, label] += probabilities[:, cluster]
                
        # Normalize again
        row_sums = result.sum(axis=1, keepdims=True)
        return result / (row_sums + 1e-10)  # Add small constant to avoid division by zero

# ---------------------- Example Usage ----------------------

def main():
    # Set your dataset folder path
    folder_path = 'Aerial_Landscapes'
    
    # Extract features using SIFT and LBP with augmentation
    # Options for feature_type: 'sift', 'lbp', or 'combined'
    X_train, y_train, X_val, y_val, class_names = extract_features_with_augmentation(
        folder_path,
        n_augmentations=3,  # Generate 3 augmented versions per image
        feature_type='combined',  # Use both SIFT and LBP features
        sample_size=300  # Sample only 10 images per class for quick testing
    )
    
    # Train SVM, KNN, and K-means models
    ml_results = train_ml_models(X_train, y_train, X_val, y_val, class_names)
    
    # Compare results
    print("\n===== MODEL COMPARISON =====")
    for name, result in ml_results.items():
        print(f"{name}: Accuracy = {result['accuracy']:.4f}")
    
    return ml_results

if __name__ == "__main__":
    # Set to True for a quick test run with minimal samples
    quick_test = True
    
    if quick_test:
        # Use minimal samples for quick testing
        ml_results = main()
    else:
        # Use the full dataset
        # Set your dataset folder path
        folder_path = 'Aerial_Landscapes'
        
        # Extract features using the full dataset
        X_train, y_train, X_val, y_val, class_names = extract_features_with_augmentation(
            folder_path,
            n_augmentations=3,
            feature_type='combined',
            sample_size=None  # Use all images
        )
        
        # Train models
        ml_results = train_ml_models(X_train, y_train, X_val, y_val, class_names)

if __name__ == "__main__":
    ml_results = main()