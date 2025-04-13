import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.manifold import TSNE
import joblib
import torch
from PIL import Image
from torch.utils.data import DataLoader
import xgboost as xgb
import catboost as cb

# import image augmentation generator
from ml_image_augmentation import AugmentationGenerator

# import feature extractir
from ml_feature_extractor import FeatureExtractor

# Get path for dataset, train, val and test
datasets_dir = Path("datasets")
train_dir = datasets_dir / "train"
val_dir = datasets_dir / "val"
test_dir = datasets_dir / "test"

# Gaussian Mixture Model-based classifier
class GMMClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, n_components=1, covariance_type='full', max_iter=100):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=42
        )
        self.cluster_to_label_map = {}
        
    def fit(self, X, y):
        # Fit GMM
        self.gmm.fit(X)
        
        # Predict cluster assignments for training data
        cluster_labels = self.gmm.predict(X)
        
        # Map each cluster to the most common class label
        for cluster in range(self.n_components):
            # Find indices of samples in this cluster
            indices = np.where(cluster_labels == cluster)[0]
            if len(indices) > 0:
                # Get the most common class label in this cluster
                most_common_label = np.bincount(y[indices]).argmax()
                self.cluster_to_label_map[cluster] = most_common_label
            else:
                # Handle empty clusters with a default value
                self.cluster_to_label_map[cluster] = 0
                
        return self
    
    def predict(self, X):
        # Predict clusters
        cluster_labels = self.gmm.predict(X)
        
        # Map clusters to class labels
        return np.array([self.cluster_to_label_map[c] for c in cluster_labels])
    
    def predict_proba(self, X):
        # Get GMM probabilities
        proba = self.gmm.predict_proba(X)
        
        # Convert to class probabilities
        n_classes = max(self.cluster_to_label_map.values()) + 1
        result = np.zeros((X.shape[0], n_classes))
        
        for cluster, label in self.cluster_to_label_map.items():
            result[:, label] += proba[:, cluster]
            
        # Normalise to ensure probabilities sum to 1
        row_sums = result.sum(axis=1, keepdims=True)
        return result / (row_sums + 1e-10)
    
# K-means based classifier that maps cluster assignments to class labels
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

# Create a Random Forest classifier
def create_random_forest(n_estimators=100, max_depth=None, max_features='sqrt', bootstrap=True):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=42,
        n_jobs=-1
    )

# Create an XG Boost classifier
def create_xgboost(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8):
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="multi:softprob",  # For multi-class classification
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'  # For multi-class classification
    )

# Create a Bagging classifier
def create_bagging(base_estimator=None, n_estimators=10, max_samples=1.0, bootstrap=True):

    return BaggingClassifier(
        base_estimator=base_estimator,
        n_estimators=n_estimators,
        max_samples=max_samples,
        bootstrap=bootstrap,
        random_state=42,
        n_jobs=-1
    )

# Create a Voting classifier
def create_voting_classifier(estimators, voting='hard', weights=None):
    return VotingClassifier(
        estimators=estimators,
        voting=voting,
        weights=weights,
        n_jobs=-1
    )

# Create a Stacking classifer
def create_stacking_classifier(estimators=None, final_estimator=None, cv=5):
    if estimators is None:
        # Default base estimators
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, random_state=42)),
            ('log', LogisticRegression(C=1.0, penalty='l1', solver='saga', random_state=42)),
            ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
        ]
    
    if final_estimator is None:
        # Default meta-estimator
        final_estimator = LogisticRegression(random_state=42)
    
    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv,
        n_jobs=-1
    )

def evaluate_unsupervised_clustering(X, y, method='kmeans', n_clusters_range=None, **kwargs):

    if n_clusters_range is None:
        n_unique_classes = len(np.unique(y))
        n_clusters_range = [
            max(2, n_unique_classes - 1),
            n_unique_classes,
            n_unique_classes + 1,
            n_unique_classes * 2
        ]
        
    best_score = -1
    best_model = None
    best_metrics = {}
    
    # For other methods, try different n_clusters
    for n_clusters in n_clusters_range:
        # Create clustering model
        if method == 'kmeans':
            cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'gmm':
            covariance_type = kwargs.get('covariance_type', 'full')
            cluster_model = GaussianMixture(
                n_components=n_clusters, 
                covariance_type=covariance_type,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Fit model and get cluster labels
        cluster_labels = cluster_model.fit_predict(X)
        
        # Calculate metrics
        try:
            silhouette = silhouette_score(X, cluster_labels)
            calinski = calinski_harabasz_score(X, cluster_labels)
            
            # Map clusters to most common ground truth labels
            mapped_labels = np.zeros_like(cluster_labels)
            for cluster in range(n_clusters):
                indices = np.where(cluster_labels == cluster)[0]
                if len(indices) > 0:
                    most_common = np.bincount(y[indices]).argmax()
                    mapped_labels[indices] = most_common
                    
            # Calculate accuracy of mapped labels
            accuracy = accuracy_score(y, mapped_labels)
            
            # Combine metrics (weighted average)
            # Normalize calinski score to 0-1 range (approximately)
            normalized_calinski = min(1.0, calinski / 1000)
            combined_score = (silhouette + normalized_calinski + accuracy) / 3
            
            if combined_score > best_score:
                best_score = combined_score
                best_model = cluster_model
                best_metrics = {
                    'n_clusters': n_clusters,
                    'silhouette': silhouette,
                    'calinski_harabasz': calinski,
                    'accuracy': accuracy
                }
                
                # Add method-specific metrics
                if method == 'kmeans':
                    best_metrics['inertia'] = cluster_model.inertia_
                elif method == 'gmm':
                    best_metrics['covariance_type'] = covariance_type
                    best_metrics['bic'] = cluster_model.bic(X)
                    best_metrics['aic'] = cluster_model.aic(X)
        
        except Exception as e:
            # Skip configurations that cause errors
            print(f"Error with {method} (n_clusters={n_clusters}): {e}")
            continue
    
    return best_model, best_metrics

# Try different hyperparameters for a model and return the best one.
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
        max(30, int(max_components * 0.75)),  # 75% of max
        max(40, int(max_components * 0.95)),  # 95% of max
        max_components,  # Maximum (100% or 100, whichever is smaller)
        None  # No PCA - use all features
    ]
    
    # Remove duplicates and sort (putting None at the end)
    pca_components = sorted([c for c in set(pca_components) if c is not None]) + [None]
    
    # Define model-specific hyperparameters
    if model_type == 'SVM':
        model_params = [
            # RBF kernel with various C values and gamma settings
            {'kernel': 'rbf', 'C': 0.1, 'gamma': 'scale'},
            {'kernel': 'rbf', 'C': 1, 'gamma': 'scale'},
            {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'},
            {'kernel': 'rbf', 'C': 100, 'gamma': 'scale'},
            {'kernel': 'rbf', 'C': 1000, 'gamma': 'scale'},
            # Add linear kernel since it sometimes performs better
            {'kernel': 'linear', 'C': 1},
            {'kernel': 'linear', 'C': 10},
            # Add polynomial kernel
            {'kernel': 'poly', 'C': 10, 'degree': 2, 'gamma': 'scale'},
        ]
    elif model_type == 'KNN':
        model_params = [
            {'n_neighbors': 1, 'weights': 'uniform'},
            {'n_neighbors': 3, 'weights': 'uniform'},
            {'n_neighbors': 5, 'weights': 'uniform'},
            {'n_neighbors': 7, 'weights': 'uniform'},
            {'n_neighbors': 1, 'weights': 'distance'},
            {'n_neighbors': 3, 'weights': 'distance'},
            {'n_neighbors': 5, 'weights': 'distance'},
            {'n_neighbors': 7, 'weights': 'distance'},
            {'n_neighbors': 9, 'weights': 'distance'},
        ]
    elif model_type == 'K-means':
        model_params = [
            {'n_clusters': len(class_names)},
            {'n_clusters': len(class_names) * 2},
            {'n_clusters': len(class_names) * 3},
            {'n_clusters': max(len(class_names) // 2, 2)},
            {'n_clusters': min(len(class_names) * 4, 30)}
        ]
    elif model_type == 'RF': 
        model_params = [
            {'n_estimators': 50, 'max_depth': None, 'max_features': 'sqrt'},
            {'n_estimators': 100, 'max_depth': None, 'max_features': 'sqrt'},
            {'n_estimators': 200, 'max_depth': None, 'max_features': 'sqrt'},
            {'n_estimators': 100, 'max_depth': 10, 'max_features': 'sqrt'},
            {'n_estimators': 100, 'max_depth': 20, 'max_features': 'sqrt'},
            {'n_estimators': 100, 'max_depth': None, 'max_features': 'log2'},
            {'n_estimators': 100, 'max_depth': None, 'max_features': None}
        ]
    elif model_type == 'XGBoost':
        model_params = [
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
            {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
            {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8},
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 1.0, 'colsample_bytree': 1.0},
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.6, 'colsample_bytree': 0.6},
            {'n_estimators': 100, 'max_depth': 7, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8},
            {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.01, 'subsample': 0.8, 'colsample_bytree': 0.8}
        ]
    elif model_type == 'CatBoost':
        model_params = [
            {'iterations': 100, 'learning_rate': 0.1, 'depth': 6, 'l2_leaf_reg': 3},
            {'iterations': 200, 'learning_rate': 0.1, 'depth': 6, 'l2_leaf_reg': 3},
            {'iterations': 100, 'learning_rate': 0.05, 'depth': 6, 'l2_leaf_reg': 3},
            {'iterations': 100, 'learning_rate': 0.1, 'depth': 6, 'l2_leaf_reg': 1},
            {'iterations': 100, 'learning_rate': 0.1, 'depth': 6, 'l2_leaf_reg': 10}
        ]
    elif model_type == 'LogisticRegression':
        model_params = [
            {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs'},
            {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'},
            {'C': 10.0, 'penalty': 'l2', 'solver': 'lbfgs'},
            {'C': 0.1, 'penalty': 'l2', 'solver': 'newton-cg'},
            {'C': 1.0, 'penalty': 'l2', 'solver': 'newton-cg'},
            {'C': 10.0, 'penalty': 'l2', 'solver': 'newton-cg'},
            {'C': 1.0, 'penalty': 'l1', 'solver': 'saga'},
            {'C': 10.0, 'penalty': 'l1', 'solver': 'saga'},
            {'C': 1.0, 'penalty': 'elasticnet', 'solver': 'saga', 'l1_ratio': 0.5}
        ]
    elif model_type == 'Stacking':
        # For stacking, we'll use a fixed set of base models and try different final estimators
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, random_state=42)),
            ('log', LogisticRegression(C=1.0, penalty='l1', solver='saga', random_state=42)),
            ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
        ]
        model_params = [
            {'estimators': base_estimators, 'final_estimator': LogisticRegression(random_state=42), 'cv': 3},
            {'estimators': base_estimators, 'final_estimator': LogisticRegression(random_state=42), 'cv': 5},
            {'estimators': base_estimators, 'final_estimator': RandomForestClassifier(n_estimators=50, random_state=42), 'cv': 3},
            {'estimators': base_estimators, 'final_estimator': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42), 'cv': 3}
        ]
    elif model_type == 'Bagging':
        model_params = [
            {'n_estimators': 10, 'max_samples': 0.8},
            {'n_estimators': 10, 'max_samples': 1.0},
            {'n_estimators': 20, 'max_samples': 0.8},
            {'n_estimators': 20, 'max_samples': 1.0}
        ]
    elif model_type == 'GMM':
        model_params = [
            {'n_components': len(class_names), 'covariance_type': 'full'},
            {'n_components': len(class_names) * 2, 'covariance_type': 'full'},
            {'n_components': len(class_names), 'covariance_type': 'tied'},
            {'n_components': len(class_names) * 2, 'covariance_type': 'tied'},
            {'n_components': len(class_names), 'covariance_type': 'diag'},
            {'n_components': len(class_names) * 2, 'covariance_type': 'diag'}
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Try combinations of PCA components and model hyperparameters
    print(f"  Testing PCA components: {pca_components}")
    
    for n_components in pca_components:
        for params in model_params:
            start_time = time.time()
            
            # Create pipeline with current hyperparameters
            pipeline_steps = []
            
            # Always start with scaling
            pipeline_steps.append(('scaler', StandardScaler()))
            
            # Add PCA if n_components is not None
            if n_components is not None:
                pipeline_steps.append(('pca', PCA(n_components=n_components)))
            
            # Add the model
            if model_type == 'SVM':
                pipeline_steps.append(('model', SVC(probability=True, random_state=42, **params)))
            elif model_type == 'KNN':
                pipeline_steps.append(('model', KNeighborsClassifier(**params)))
            elif model_type == 'K-means':
                pipeline_steps.append(('model', KMeansClassifier(**params)))
            elif model_type == 'RF':
                pipeline_steps.append(('model', RandomForestClassifier(random_state=42, **params)))
            elif model_type == 'XGBoost':
                pipeline_steps.append(('model', xgb.XGBClassifier(
                    random_state=42, 
                    use_label_encoder=False, 
                    eval_metric='mlogloss',
                    objective="multi:softprob",
                    n_jobs=-1,
                    **params
                )))
            elif model_type == 'CatBoost':
                pipeline_steps.append(('model', cb.CatBoostClassifier(
                    random_seed=42,
                    verbose=0,
                    thread_count=-1,
                    **params
                )))
            elif model_type == 'LogisticRegression':
                pipeline_steps.append(('model', LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    multi_class='multinomial',
                    n_jobs=-1,
                    **params
                )))
            elif model_type == 'Stacking':
                pipeline_steps.append(('model', create_stacking_classifier(**params)))
            elif model_type == 'Bagging':
                # Use SVC as base estimator for bagging
                svc = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42)
                pipeline_steps.append(('model', BaggingClassifier(base_estimator=svc, random_state=42, **params)))
            elif model_type == 'GMM':
                pipeline_steps.append(('model', GMMClassifier(**params)))
            
            pipeline = Pipeline(pipeline_steps)
            
            # Train model
            try:
                pipeline.fit(X_train, y_train)
                
                # Evaluate on validation set
                y_pred = pipeline.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                
                end_time = time.time()
                
                pca_str = f"PCA={n_components}" if n_components is not None else "No PCA"
                print(f"  {model_type} {params}, {pca_str}: Val Acc = {accuracy:.4f} (Time: {end_time - start_time:.2f}s)")
                
                # Check if this is the best model so far
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {'model_params': params, 'pca_components': n_components}
                    best_model = pipeline
            except Exception as e:
                print(f"  Error with {model_type} {params}, {pca_str}: {e}")
                continue
    
    if best_model is None:
        print(f"Warning: Could not find a working model for {model_type}. Using default parameters.")
        
        # Create a default model as fallback
        if model_type == 'XGBoost':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', xgb.XGBClassifier(
                    random_state=42, 
                    use_label_encoder=False, 
                    eval_metric='mlogloss',
                    n_jobs=-1
                ))
            ])
        elif model_type == 'CatBoost':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', cb.CatBoostClassifier(
                    random_seed=42,
                    verbose=0,
                    thread_count=-1
                ))
            ])
        elif model_type == 'LogisticRegression':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    multi_class='multinomial',
                    n_jobs=-1
                ))
            ])
        elif model_type == 'Stacking':
            base_estimators = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('svm', SVC(probability=True, random_state=42)),
                ('log', LogisticRegression(C=1.0, penalty='l1', solver='saga', random_state=42)),
                ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
            ]
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=LogisticRegression(random_state=42),
                    cv=3,
                    n_jobs=-1
                ))
            ])
        elif model_type == 'SVM':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVC(probability=True, random_state=42))
            ])
        elif model_type == 'KNN':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', KNeighborsClassifier(n_neighbors=5))
            ])
        elif model_type == 'RF':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(random_state=42))
            ])
        else:
            # Use random forest as a fallback for any other model
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(random_state=42))
            ])
            
        pipeline.fit(X_train, y_train)
        best_model = pipeline
        best_params = {'model_params': {}, 'pca_components': None}
        best_accuracy = 0.0
    else:
        print(f"Best {model_type} hyperparameters: {best_params}")
        print(f"Best {model_type} validation accuracy: {best_accuracy:.4f}")
    
    return best_model, best_params, best_accuracy

# Train a final model on combined training and validation data
def train_final_model(X_combined, y_combined, model_type, best_params, pca_n_components):

    print(f"\nTraining final {model_type} model with best hyperparameters on combined data...")
    
    # Extract model-specific params and PCA components
    model_params = best_params['model_params']
    n_components = best_params['pca_components']
    
    pca_str = f"{n_components} PCA components" if n_components is not None else "No PCA"
    print(f"  Using {pca_str} and model parameters: {model_params}")
    
    # Create pipeline with best hyperparameters
    pipeline_steps = []
    
    # Always start with scaling
    pipeline_steps.append(('scaler', StandardScaler()))
    
    # Add PCA if n_components is not None
    if n_components is not None:
        pipeline_steps.append(('pca', PCA(n_components=n_components)))
    
    # Add the model
    if model_type == 'SVM':
        pipeline_steps.append(('model', SVC(probability=True, random_state=42, **model_params)))
    elif model_type == 'KNN':
        pipeline_steps.append(('model', KNeighborsClassifier(**model_params)))
    elif model_type == 'K-means':
        pipeline_steps.append(('model', KMeansClassifier(**model_params)))
    elif model_type == 'RF':
        pipeline_steps.append(('model', RandomForestClassifier(random_state=42, **model_params)))
    elif model_type == 'XGBoost':
        pipeline_steps.append(('model', xgb.XGBClassifier(
            random_state=42, 
            use_label_encoder=False, 
            eval_metric='mlogloss',
            objective="multi:softprob",
            n_jobs=-1,
            **model_params
        )))
    elif model_type == 'CatBoost':
        pipeline_steps.append(('model', cb.CatBoostClassifier(
            random_seed=42,
            verbose=0,
            thread_count=-1,
            **model_params
        )))
    elif model_type == 'LogisticRegression':
        pipeline_steps.append(('model', LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='multinomial',
            n_jobs=-1,
            **model_params
        )))
    elif model_type == 'Stacking':
        pipeline_steps.append(('model', create_stacking_classifier(**model_params)))
    elif model_type == 'Bagging':
        # Use SVC as base estimator for bagging
        svc = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42)
        pipeline_steps.append(('model', BaggingClassifier(base_estimator=svc, random_state=42, **model_params)))
    elif model_type == 'GMM':
        pipeline_steps.append(('model', GMMClassifier(**model_params)))
    
    pipeline = Pipeline(pipeline_steps)
    
    # Train model on combined data
    pipeline.fit(X_combined, y_combined)
    
    return pipeline

# Evaluate a model on test data
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

def main(sample_size=None, feature_type='sift', n_augmentations=5, batch_size=32):

    # Check if dataset exists
    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            "Dataset directories not found. Please run split_dataset.py first."
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
    
    print("\nStep 1: Creating dataloaders...")
    # Create dataloaders for training and validation sets
    # Pass specific sample sizes for train and validation datasets
    train_dataset, val_dataset = augmentor.create_datasets(
        train_dir, val_dir, sample_size=train_sample_size, val_sample_size=val_sample_size
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create a data loader for test set
    test_dataset = augmentor.create_datasets(
        test_dir, test_dir, sample_size=test_sample_size
    )[0]  # Just use the first returned dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Get class names from the train dataset
    class_indices = {idx: classname for classname, idx in train_loader.dataset.class_to_idx.items()}
    print(f"Classes: {class_indices}")
    
    # Step 2: Create feature extractor and codebook
    feature_extractor = FeatureExtractor(feature_type=feature_type)
    
    print("\nStep 2: Creating codebook for bag of visual words...")
    if feature_type in ['sift', 'combined']:
        feature_extractor.create_codebook(train_loader, n_clusters=200)
        # The codebook is stored in the feature_extractor object itself
    
    # Step 3: Extract features from training images
    print("\nStep 3: Extracting features from training images...")
    X_train, y_train = feature_extractor.extract_features_batch(train_loader)
    
    # Step 4: Extract features from validation images
    print("\nStep 4: Extracting features from validation images...")
    X_val, y_val = feature_extractor.extract_features_batch(val_loader)
    
    print("\nFeature extraction complete:")
    print(f"Train: {X_train.shape} features, {y_train.shape} labels")
    print(f"Validation: {X_val.shape} features, {y_val.shape} labels")
    
    # Step 5: Try different hyperparameters and select the best for each model
    pca_n_components = min(100, X_train.shape[1])  # Limit to 100 components or feature dimension
    
    # Try hyperparameters for each model type and select the best
    best_models = {}
    best_params = {}
    
    print("\nStep 5: Finding optimal hyperparameters...")
    
    # List of models to try
    model_types = [
        # Supervised methods
        'SVM', 'KNN', 'RF', 'XGBoost', 'CatBoost', 'LogisticRegression', 'Stacking', 'Bagging',
        # Unsupervised methods
        'K-means', 'GMM'
    ]
    
    for model_type in model_types:
        print(f"\nTrying hyperparameters for {model_type}...")
        try:
            best_model, params, accuracy = try_hyperparameters(
                X_train, y_train, X_val, y_val, model_type, class_indices, pca_n_components
            )
            best_models[model_type] = best_model
            best_params[model_type] = params
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            print(f"Skipping {model_type} model")
    
    # Step 6: Create and evaluate ensemble models
    print("\nStep 6: Creating ensemble models...")
    
    # Create a voting ensemble from the best individual models
    voting_classifiers = []
    for model_type, model in best_models.items():
        # Only include supervised classifiers with decent performance
        if model_type in ['SVM', 'KNN', 'RF', 'XGBoost', 'CatBoost', 'LogisticRegression'] and hasattr(model, 'predict_proba'):
            voting_classifiers.append((model_type, model))
    
    if len(voting_classifiers) >= 2:
        # Create soft voting ensemble
        voting_ensemble = VotingClassifier(
            estimators=voting_classifiers,
            voting='soft'
        )
        
        # Train on combined data
        print("Training Voting Ensemble model...")
        voting_ensemble.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = voting_ensemble.predict(X_val)
        voting_acc = accuracy_score(y_val, y_pred)
        print(f"Voting Ensemble Validation Accuracy: {voting_acc:.4f}")
        
        # Add to best models
        best_models['Voting'] = voting_ensemble
        best_params['Voting'] = {'models': [name for name, _ in voting_classifiers]}
    else:
        print("Not enough models for a voting ensemble")
    
    # Step 7: Combine training and validation data
    print("\nStep 7: Combining training and validation data for final model training...")
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.concatenate((y_train, y_val))
    print(f"Combined data shape: {X_combined.shape}")
    
    # Step 8: Train final models on combined data with best hyperparameters
    final_models = {}
    
    print("\nStep 8: Training final models with best hyperparameters...")
    for model_type, model in best_models.items():
        if model_type in ['Voting', 'Custom_Ensemble']:
            # Retrain ensemble models on combined data
            model.fit(X_combined, y_combined)
            final_models[model_type] = model
        elif model_type in best_params:
            # Train regular models using train_final_model function
            final_models[model_type] = train_final_model(
                X_combined, y_combined, model_type, best_params[model_type], pca_n_components
            )
        # Save the final model
        try:
            joblib.dump(final_models[model_type], f'final_{model_type.lower()}_model.pkl')
        except Exception as e:
            print(f"Error saving {model_type} model: {e}")
    
    # Step 9: Extract features from test images
    print("\nStep 9: Extracting features from test images...")
    X_test, y_test = feature_extractor.extract_features_batch(test_loader)
    
    # Step 10: Evaluate final models on test set
    print("\nStep 10: Evaluating final models on test set...")
    results = {}
    
    for model_type, model in final_models.items():
        try:
            accuracy, report, conf_matrix = evaluate_model(model, X_test, y_test, class_indices, model_type)
            
            results[model_type] = {
                'model': model,
                'params': best_params.get(model_type, {}),
                'accuracy': accuracy,
                'report': report,
                'conf_matrix': conf_matrix
            }
        except Exception as e:
            print(f"Error evaluating {model_type}: {e}")
            continue
    
    # Step 11: Compare final results
    print("\n===== FINAL MODEL COMPARISON (TEST SET) =====")
    for model_type, result in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"{model_type}: Test Accuracy = {result['accuracy']:.4f}")
        if isinstance(result['params'], dict) and 'model_params' in result['params']:
            print(f"  Parameters: {result['params']['model_params']}")
        else:
            print(f"  Parameters: {result['params']}")
    
    # Step 12: Visualize feature space using t-SNE (optional)
    try:
        print("\nStep 12: Visualizing feature space with t-SNE...")
        # Apply PCA first to reduce dimensionality (for faster t-SNE)
        pca = PCA(n_components=min(50, X_test.shape[1]))
        X_pca = pca.fit_transform(X_test)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_pca)
        
        # Plot t-SNE visualization
        plt.figure(figsize=(12, 10))
        for class_idx in np.unique(y_test):
            # Get data points for this class
            mask = y_test == class_idx
            plt.scatter(
                X_tsne[mask, 0], X_tsne[mask, 1],
                label=class_indices.get(class_idx, f'Class {class_idx}'),
                alpha=0.7
            )
        
        plt.title('t-SNE Visualization of Feature Space')
        plt.legend()
        plt.savefig('feature_space_tsne.png', dpi=300)
        plt.show()
    except Exception as e:
        print(f"Error creating t-SNE visualization: {e}")
    
    return results

# ---------------------- Script Execution ----------------------

if __name__ == "__main__":
    # Set to True for quick testing with a small sample
    quick_test = True
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if quick_test:
        # Calculate sample sizes based on the original 7:1.5:1.5 ratio
        total_sample_size = 100  # Total samples per class across all splits
        train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
        
        # Define a sample size dictionary to pass to the main function
        sample_sizes = {
            'train': int(total_sample_size * train_ratio),  # 70 samples
            'val': max(1, int(total_sample_size * val_ratio)),  # ~15 samples
            'test': max(1, int(total_sample_size * test_ratio))  # ~15 samples
        }
        
        print("QUICK TEST MODE: Using proportional samples for testing")
        print(f"Sample sizes per class - Train: {sample_sizes['train']}, Val: {sample_sizes['val']}, Test: {sample_sizes['test']}")
        results = main(sample_size=sample_sizes, n_augmentations=2, batch_size=16)
    else:
        print("FULL DATASET MODE: Using all available images")
        results = main(sample_size=None, n_augmentations=3, batch_size=32)
    import csv
    csv_file = 'file.csv'
    # Writing to CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', 'Value'])
    	
        # Write data
        for key, value in results.items():
            writer.writerow([key, value])

    # Print final ranking of models
    print("\nFINAL MODEL RANKING BY TEST ACCURACY:")
    for i, (model_type, result) in enumerate(sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True), 1):
        print(f"{i}. {model_type}: {result['accuracy']:.4f}")