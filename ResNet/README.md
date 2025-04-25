# ResNet Image Classification Project

This project explores image classification using different ResNet backbones (ResNet-18, ResNet-50, ResNet-101) and various classification heads, including traditional classifiers and fine-tuning strategies. We also investigate robustness under long-tail distributions and test-time perturbations.

##  Project Structure

```
.
├── train_and_test_model.py           # ResNet fine-tuning with only fc layer unfrozen
├── ResNet_Classifier.py              # ResNet feature extraction + SVM/KNN/MLP/Proto classifiers
├── Ablation_study.ipynb              # Ablation study on ResNet models and classifiers
├── ResNet_longtail_experienment.ipynb # Long-tail distribution and re-weighting/re-sampling
├── ResNet_Model.ipynb                # Main notebook to run training and evaluation
```

##  Main Components

### 1. `train_and_test_model.py`
- Implements fine-tuning where only the final fully connected (fc) layer of ResNet is trained.
- Supports ResNet-18, ResNet-50, and ResNet-101.
- Includes early stopping and confusion matrix evaluation.

### 2. `ResNet_Classifier.py`
- Uses pretrained ResNet backbones to extract features (fc replaced by `nn.Identity`).
- Supports multiple classifiers:
  - SVM (Support Vector Machine)
  - KNN (k-Nearest Neighbors)
  - MLP (Multi-Layer Perceptron)
  - Prototypical Network
- Performs accuracy and confusion matrix evaluation.

### 3. `Ablation_study.ipynb`
- Systematic experiments comparing:
  - Different ResNet depths
  - Fine-tuning vs. frozen feature extractors
  - Effects of various classifiers on performance

### 4. `ResNet_longtail_experienment.ipynb`
- Investigates class imbalance handling:
  - Re-sampling
  - Class re-weighting
- Evaluates model robustness on skewed data distributions.

### 5. `ResNet_Model.ipynb`
- Central notebook to run standard training/inference pipeline.
- Uses `train_and_test_model.py` and `ResNet_Classifier.py` as backend.

##  Experiments

- **Backbone Comparison**: ResNet-18, 50, 101 under same training setting
- **Classification Heads**: SVM performs best in most settings
- **Long-Tail Handling**: Evaluated under re-sampling and re-weighting
- **Robustness Test**: Models tested under various perturbation levels (mild, medium, strong)
