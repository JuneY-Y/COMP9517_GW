# Aerial Image Classification Project

This project implements a comprehensive machine learning pipeline for aerial image classification using traditional computer vision techniques combined with multiple machine learning algorithms. The focus is on extracting rich features from aerial imagery and evaluating different classification approaches.

## Project Structure

```
.
├── ml_image_augmentation.py     # Image enhancement and augmentation techniques
├── ml_feature_extractor.py      # Main feature extraction pipeline
├── ml_color_sift.py             # SIFT feature extraction in multiple color spaces
├── ml_color_lbp.py              # Multi-scale LBP feature extraction
├── ml_modelling.py              # ML model training, evaluation and ensemble creation
```

## Main Components

### 1. `ml_image_augmentation.py`
- Implements contrast enhancement using CLAHE
- Applies image sharpening via unsharp masking
- Creates custom augmentations (occlusions, color jitters, rotations)
- Provides dataset and dataloader creation utilities

### 2. `ml_feature_extractor.py`
- Combines multiple feature extraction techniques
- Manages batch processing for datasets
- Handles feature normalization and codebook creation

### 3. `ml_color_sift.py`
- Extracts SIFT features in RGB, opponent, and HSV color spaces
- Creates Bag of Visual Words representations
- Computes visual word histograms from local features

### 4. `ml_color_lbp.py`
- Extracts Local Binary Pattern features at multiple scales
- Processes features across color channels
- Implements opponent color space transformations

### 5. `ml_modelling.py`
- Trains and evaluates multiple classifiers:
  - Traditional (SVM, KNN, Random Forests)
  - Boosting (XGBoost, CatBoost)
  - Ensemble (Stacking, Bagging, Voting)
  - Unsupervised (K-means, GMM)
- Performs hyperparameter optimization
- Creates feature space visualizations
- Evaluates model performance metrics

## Experiments

- **Feature Comparison**: Combined Color SIFT + LBP vs. individual feature types
- **Classifier Evaluation**: Systematic comparison across multiple algorithms
- **Ensemble Approaches**: Voting and stacking strategies for improved performance
- **Feature Visualization**: t-SNE embedding of the feature space

## Usage

Run the main pipeline with:
```python
from ml_modelling import main

# Full dataset
results = main(sample_size=None, feature_type='combined', batch_size=32)

# Quick test
results = main(sample_size=100, feature_type='combined', batch_size=16)
```

The model requires a dataset organized in train/val/test folders with class subfolders.
