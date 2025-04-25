# Machine Learning

This is an implementation of traditional machine learning methods. The focus is on extracting rich features from aerial imagery and evaluating different classification approaches.

## Project Structure

```
.
├── ml_image_augmentation.py     # Image enhancement and augmentation techniques
├── ml_feature_extractor.py      # Main feature extraction pipeline
├── ml_color_sift.py             # SIFT feature extraction in multiple color spaces
├── ml_color_lbp.py              # Multi-scale LBP feature extraction
├── ml_modelling.py              # ML model training, evaluation and ensemble creation
├── ml_modelling_imbalance.py    # Adapts ML modelling to handle class imbalance 
```

## Main Components

### 1. `ml_image_augmentation.py`
- Implements contrast enhancement using CLAHE
- Applies image sharpening via unsharp masking
- Creates custom augmentations (brightness, rotations, flips, occlusions)
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
  - Traditional (SVM, Logistic Regression, KNN, Random Forests)
  - Boosting (XGBoost, CatBoost)
  - Ensemble (Stacking, Voting)
  - Unsupervised (K-means, GMM)
- Performs hyperparameter optimization
- Evaluates model performance metrics
- Adaptability to data perturbation experiments

### 6. `ml_modelling_imbalance.py`

- Adapts `ml_modelling.py` to handle class imbalance

## Usage

Run the main pipeline with:
```python
from ml_modelling import main
# or from ml_modelling_imbalance import main

# Full dataset
results = main(sample_size=None, feature_type='combined', batch_size=32)

# Quick test
results = main(sample_size=100, feature_type='combined', batch_size=16)
```

The model requires a dataset organized in train/val/test folders with class subfolders.
