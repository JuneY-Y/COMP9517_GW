import cv2
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
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

# Import from image_augmentation module
from image_augmentation import AugmentationGenerator

# Get path for dataset, train, val and test
datasets_dir = Path("datasets")
train_dir = datasets_dir / "train"
val_dir = datasets_dir / "val"
test_dir = datasets_dir / "test"
        
# Function to extract an image's sift features
def extract_sift_features(image, n_features=100):

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=n_features)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # If no keypoints were found, return empty array with correct shape
    if descriptors is None:
        # SIFT descriptors are 128-dimensional
        return np.zeros((0, 128)) 
    
    return descriptors

# Function to extract image features using LBP
def extract_lbp_features(image, n_points=24, radius=8, method='uniform'):

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method)
    
    # Compute histogram of LBP
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist

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
    
    # Assign each descriptor to the nearest visual word
    for descriptor in descriptors:
        distances = np.sqrt(((codebook - descriptor)**2).sum(axis=1))
        nearest_word = np.argmin(distances)
        hist[nearest_word] += 1
    
    # Normalize histogram
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    
    return hist

# Function to extract SIFT features from a list of images for codebook creation
def extract_features_from_images(images, feature_type='combined'):

    features = []
    sift_descriptors = []
    
    for img in images:
        # Extract SIFT descriptors for codebook creation
        if feature_type == 'sift' or feature_type == 'combined':
            descriptors = extract_sift_features(img)
            if descriptors.shape[0] > 0:
                sift_descriptors.append(descriptors)
    
    return features, sift_descriptors

# Function to extract SIFT images using a pre-computed codebook
def extract_features_with_codebook(images, codebook, feature_type='combined'):

    features = []
    fixed_lbp_size = None
    
    for img in images:
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
            feature_vector = bovw_features
        elif feature_type == 'lbp':
            feature_vector = lbp_features
        else:  # combined
            feature_vector = np.concatenate([bovw_features, lbp_features])
        
        features.append(feature_vector)
    
    return features

class KMeansClassifier(BaseEstimator, ClassifierMixin):

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
        
        # Normalise to get probabilities
        probabilities = similarities / similarities.sum(axis=1, keepdims=True)
        
        # Reorder to match class labels
        n_classes = max(self.cluster_to_label_map.values()) + 1
        result = np.zeros((X.shape[0], n_classes))
        
        for cluster, label in self.cluster_to_label_map.items():
            if label >= 0:  # Skip -1 labels (empty clusters)
                result[:, label] += probabilities[:, cluster]
                
        # Normalise again
        row_sums = result.sum(axis=1, keepdims=True)
        return result / (row_sums + 1e-10)  # Add small constant to avoid division by zero

def try_hyperparameters(X_train, y_train, X_val, y_val, model_type, class_names, pca_n_components):
    
    best_model = None
    best_params = None
    best_accuracy = 0.0
    
    # Define PCA component options for tuning
    # Use smaller values for small datasets, larger values for larger feature spaces
    max_components = min(X_train.shape[1], 100)  # Cap at original dimension or 100
    pca_components = [
        max(5, int(max_components * 0.1)),   # 10% of max
        max(10, int(max_components * 0.25)),  # 25% of max
        max(20, int(max_components * 0.5)),   # 50% of max
        max(25, int(max_components * 0.75)),  # 75% of max
        max(30, int(max_components * 0.95)),  # 95% of max
        max_components  # Maximum (100% or 100, whichever is smaller)
    ]
    
    # Remove duplicates and sort
    pca_components = sorted(list(set(pca_components)))
    
    # Define model-specific hyperparameters
    if model_type == 'SVM':
        model_params = [
            # RBF kernel with various C values and gamma settings
            {'kernel': 'rbf', 'C': 0.1, 'gamma': 'scale'},
            {'kernel': 'rbf', 'C': 1, 'gamma': 'scale'},
            {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'},
            {'kernel': 'rbf', 'C': 100, 'gamma': 'scale'},
            {'kernel': 'rbf', 'C': 1000, 'gamma': 'scale'},
            
            # The below combinations are grayed out for consistent underperformance
            
            # # RBF kernel with explicit gamma values
            # {'kernel': 'rbf', 'C': 10, 'gamma': 0.001},
            # {'kernel': 'rbf', 'C': 10, 'gamma': 0.01},
            # {'kernel': 'rbf', 'C': 10, 'gamma': 0.1},
            # # Linear kernel with various C values
            # {'kernel': 'linear', 'C': 0.1},
            # {'kernel': 'linear', 'C': 1},
            # # Computations with larger C values such as 10 or 100 are prohibitively expensive
            # # Polynomial kernel with various settings
            # {'kernel': 'poly', 'C': 1, 'degree': 2, 'gamma': 'scale'},
            # {'kernel': 'poly', 'C': 10, 'degree': 2, 'gamma': 'scale'},
            # {'kernel': 'poly', 'C': 10, 'degree': 3, 'gamma': 'scale'},
            # {'kernel': 'poly', 'C': 100, 'degree': 2, 'gamma': 'scale'},
            # # Sigmoid kernel with various C values
            # {'kernel': 'sigmoid', 'C': 1, 'gamma': 'scale'},
            # {'kernel': 'sigmoid', 'C': 10, 'gamma': 'scale'},
            # {'kernel': 'sigmoid', 'C': 100, 'gamma': 'scale'},
            # {'kernel': 'sigmoid', 'C': 1000, 'gamma': 'scale'}
        ]
    elif model_type == 'KNN':
        model_params = [
            {'n_neighbors': 3, 'weights': 'uniform'},
            {'n_neighbors': 5, 'weights': 'uniform'},
            {'n_neighbors': 7, 'weights': 'uniform'},
            {'n_neighbors': 3, 'weights': 'distance'},
            {'n_neighbors': 5, 'weights': 'distance'},
            {'n_neighbors': 7, 'weights': 'distance'}
        ]
    elif model_type == 'K-means':
        model_params = [
            {'n_clusters': len(class_names)},
            {'n_clusters': len(class_names) * 2},
            {'n_clusters': max(len(class_names) // 2, 2)},
            {'n_clusters': min(len(class_names) * 3, 20)}
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Try combinations of PCA components and model hyperparameters
    print(f"  Testing PCA components: {pca_components}")
    
    for n_components in pca_components:
        for params in model_params:
            start_time = time.time()
            
            # Create pipeline with current hyperparameters
            if model_type == 'SVM':
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=n_components)),
                    ('model', SVC(probability=True, random_state=42, **params))
                ])
            elif model_type == 'KNN':
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=n_components)),
                    ('model', KNeighborsClassifier(**params))
                ])
            elif model_type == 'K-means':
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=n_components)),
                    ('model', KMeansClassifier(**params))
                ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = pipeline.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            end_time = time.time()
            
            print(f"  {model_type} {params}, PCA components={n_components}: Val Acc = {accuracy:.4f} (Time: {end_time - start_time:.2f}s)")
            
            # Check if this is the best model so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'model_params': params, 'pca_components': n_components}
                best_model = pipeline
    
    print(f"Best {model_type} hyperparameters: {best_params}")
    print(f"Best {model_type} validation accuracy: {best_accuracy:.4f}")
    
    return best_model, best_params, best_accuracy

def train_final_model(X_combined, y_combined, model_type, best_params, pca_n_components):
    
    print(f"\nTraining final {model_type} model with best hyperparameters on combined data...")
    
    # Extract model-specific params and PCA components
    model_params = best_params['model_params']
    n_components = best_params['pca_components']
    
    print(f"  Using {n_components} PCA components and model parameters: {model_params}")
    
    # Create pipeline with best hyperparameters
    if model_type == 'SVM':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components)),
            ('model', SVC(probability=True, random_state=42, **model_params))
        ])
    elif model_type == 'KNN':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components)),
            ('model', KNeighborsClassifier(**model_params))
        ])
    elif model_type == 'K-means':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components)),
            ('model', KMeansClassifier(**model_params))
        ])
    
    # Train model on combined data
    pipeline.fit(X_combined, y_combined)
    
    return pipeline

def evaluate_model(model, X_test, y_test, class_names, model_type):

    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=list(class_names.values()))
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\n{model_type} Test Accuracy: {accuracy:.4f}")
    print(f"{model_type} Classification Report:\n{report}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_type} (Test Set)')
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
    
    return accuracy, report, conf_matrix

# ---------------------- Main Function ----------------------

def main(sample_size=None, feature_type='combined', n_augmentations=3):

    # Check if dataset exists
    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            "Dataset directories not found. Please run split_dataset.py first: "
            "python split_dataset.py"
        )
    
    # Create augmentation generator
    augmentor = AugmentationGenerator()
    
    # Handle different sample size formats
    train_sample_size = sample_size
    val_sample_size = sample_size
    test_sample_size = sample_size
    
    # If sample_size is a dictionary, extract the specific sample sizes
    if isinstance(sample_size, dict):
        train_sample_size = sample_size.get('train', None)
        val_sample_size = sample_size.get('val', None)
        test_sample_size = sample_size.get('test', None)
    
    # Step 1: Load and augment training images
    print("\nStep 1: Loading and augmenting training images...")
    train_images, train_labels, class_indices = augmentor.load_and_augment_images(
        train_dir, 
        n_augmentations=n_augmentations,
        sample_size=train_sample_size
    )
    print(f"Loaded {len(train_images)} images (including augmentations) from {len(class_indices)} classes")
    
    # Step 2: Extract SIFT descriptors for codebook creation
    print("\nStep 2: Extracting SIFT descriptors for codebook creation...")
    sift_descriptors = []
    for img in train_images:
        descriptors = extract_sift_features(img)
        if descriptors.shape[0] > 0:
            sift_descriptors.append(descriptors)
    
    # Step 3: Create bag of visual words codebook
    print("\nStep 3: Creating bag of visual words codebook...")
    codebook = create_bag_of_visual_words(sift_descriptors, n_clusters=100)
    
    # Step 4: Extract features from training images
    print("\nStep 4: Extracting features from training images...")
    X_train_features = extract_features_with_codebook(train_images, codebook, feature_type)
    y_train = np.array(train_labels)
    
    # Step 5: Load validation images (no augmentation)
    print("\nStep 5: Loading validation images...")
    val_images, val_labels, _ = augmentor.load_and_augment_images(
        val_dir, 
        n_augmentations=0,  # No augmentation for validation
        sample_size=val_sample_size
    )
    print(f"Loaded {len(val_images)} validation images")
    
    # Step 6: Extract features from validation images
    print("\nStep 6: Extracting features from validation images...")
    X_val_features = extract_features_with_codebook(val_images, codebook, feature_type)
    y_val = np.array(val_labels)
    
    # Convert to numpy arrays with consistent shapes
    X_train = np.array(X_train_features)
    X_val = np.array(X_val_features)
    
    print("\nFeature extraction complete:")
    print(f"Train: {X_train.shape} features, {y_train.shape} labels")
    print(f"Validation: {X_val.shape} features, {y_val.shape} labels")
    
    # Reverse class indices for reporting
    class_names = {v: k for k, v in class_indices.items()}
    
    # Step 7: Try different hyperparameters and select the best for each model
    pca_n_components = min(100, X_train.shape[1])  # Limit to 100 components or feature dimension
    
    # Try hyperparameters for each model type and select the best
    best_models = {}
    best_params = {}
    
    for model_type in ['SVM', 'KNN', 'K-means']:
        best_model, params, accuracy = try_hyperparameters(
            X_train, y_train, X_val, y_val, model_type, class_names, pca_n_components
        )
        best_models[model_type] = best_model
        best_params[model_type] = params
    
    # Step 8: Combine training and validation data
    print("\nStep 8: Combining training and validation data for final model training...")
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.concatenate((y_train, y_val))
    print(f"Combined data shape: {X_combined.shape}")
    
    # Step 9: Train final models on combined data with best hyperparameters
    final_models = {}
    
    for model_type in ['SVM', 'KNN', 'K-means']:
        final_models[model_type] = train_final_model(
            X_combined, y_combined, model_type, best_params[model_type], pca_n_components
        )
        
        # Save the final model
        joblib.dump(final_models[model_type], f'final_{model_type.lower()}_model.pkl')
    
    # Step 10: Load test images
    print("\nStep 10: Loading test images...")
    test_images, test_labels, _ = augmentor.load_and_augment_images(
        test_dir, 
        n_augmentations=0,  # No augmentation for test
        sample_size=test_sample_size
    )
    print(f"Loaded {len(test_images)} test images")
    
    # Step 11: Extract features from test images
    print("\nStep 11: Extracting features from test images...")
    X_test_features = extract_features_with_codebook(test_images, codebook, feature_type)
    y_test = np.array(test_labels)
    X_test = np.array(X_test_features)
    
    # Step 12: Evaluate final models on test set
    print("\nStep 12: Evaluating final models on test set...")
    results = {}
    
    for model_type, model in final_models.items():
        accuracy, report, conf_matrix = evaluate_model(model, X_test, y_test, class_names, model_type)
        
        results[model_type] = {
            'model': model,
            'params': best_params[model_type],
            'accuracy': accuracy,
            'report': report,
            'conf_matrix': conf_matrix
        }
    
    # Step 13: Compare final results
    print("\n===== FINAL MODEL COMPARISON (TEST SET) =====")
    for model_type, result in results.items():
        print(f"{model_type}: Test Accuracy = {result['accuracy']:.4f}, Best Parameters: {result['params']}")
    

# ---------------------- Script Execution ----------------------

if __name__ == "__main__":
    # Set to True for quick testing with a small sample
    quick_test = True
    
    if quick_test:
        # Calculate sample sizes based on the original 7:1.5:1.5 ratio
        total_sample_size = 100  # Total samples per class across all splits
        train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
        
        # Define a sample size dictionary to pass to the main function
        sample_sizes = {
            'train': int(total_sample_size * train_ratio),  # 7 samples
            'val': max(1, int(total_sample_size * val_ratio)),  # ~1-2 samples
            'test': max(1, int(total_sample_size * test_ratio))  # ~1-2 samples
        }
        
        print("QUICK TEST MODE: Using proportional samples for testing")
        print(f"Sample sizes per class - Train: {sample_sizes['train']}, Val: {sample_sizes['val']}, Test: {sample_sizes['test']}")
        results = main(sample_size=sample_sizes, n_augmentations=2)
    else:
        print("FULL DATASET MODE: Using all available images")
        results = main(sample_size=None, n_augmentations=3)
    
