# YOLOv8 Aerial Image Classifier 🌍
build by JiamingYang

This project uses **YOLOv8's classification mode** to classify aerial landscape images (e.g., Beach, Forest, City).

timeline: updated in Apr 3th:
# 🔬 Ablation Study Design for YOLOv8 Classification

This project investigates the impact of key architectural and training components in the YOLOv8 classification model by conducting a series of ablation experiments. We aim to assess how different modules contribute to model performance, efficiency, and overall training behavior.

---

## 1️⃣ Backbone Replacement: C2f → C3 (YOLOv5-style)

- **Objective**: Evaluate the performance difference between YOLOv8's lightweight `C2f` module and YOLOv5's classic `C3` module in the backbone.
- **Modifications**:
  - Replace all `C2f` blocks with `C3` blocks while keeping the rest of the architecture unchanged.
- **Metrics Tracked**:
  - Top-1 and Top-5 classification accuracy
  - Training and validation loss
  - Total training time per epoch
  - Parameter count and model size

---

## 2️⃣ Optimizer Replacement: AdamW → SGD

- **Objective**: Compare the performance and convergence characteristics of two popular optimizers: `AdamW` (default) and `SGD`.
- **Modifications**:
  - Use identical network structure and training schedules, switching only the optimizer.
- **Metrics Tracked**:
  - Accuracy and loss over epochs
  - Training stability and convergence rate
  - Final saved model performance

---

## 3️⃣ Feature Aggregation Module: SPPF → SPP

- **Objective**: Investigate the effect of replacing the modern `SPPF` (stacked 3×3 pooling) module with the traditional `SPP` module (multi-scale pooling using larger kernels such as 5×5, 9×9, 13×13).
- **Modifications**:
  - Replace the `SPPF` layer with a custom `SPP` configuration.
  - Adjust internal convolutional kernel sizes accordingly.
- **Metrics Tracked**:
  - Classification accuracy
  - Per-epoch training time
  - Parameter count and computational complexity (FLOPs)

---

## ✅ Evaluation Metrics (Unified for All Experiments)

| Metric             | Description                                |
|--------------------|--------------------------------------------|
| **Top-1 Accuracy** | Main classification accuracy                |
| **Top-5 Accuracy** | Accuracy within top 5 predicted classes     |
| **Train/Eval Loss**| Loss curves used for convergence analysis   |
| **Params (M)**     | Total number of model parameters in millions|
| **Epoch Time (s)** | Average training time per epoch             |
| **Model Size (MB)**| Final size of the `.pt` weights file        |

---

## 🧪 Suggested Reporting Format

Include comparisons between variants in tables and graphs to clearly illustrate performance trade-offs. Example:

| Model Variant       | Backbone | Optimizer | Head | Top-1 Acc | Epoch Time (s) | Params (M) | Notes          |
|---------------------|----------|-----------|------|------------|----------------|-------------|----------------|
| YOLOv8n-cls         | C2f      | AdamW     | SPPF | 88.2%      | 32.5           | 5.1         | Baseline       |
| YOLOv8n-c3-cls      | C3       | AdamW     | SPPF | 88.7%      | 36.0           | 6.2         | Backbone test  |
| YOLOv8n-c2f-sgd     | C2f      | SGD       | SPPF | 87.5%      | 33.8           | 5.1         | Optimizer test |
| YOLOv8n-c2f-adamw-spp | C2f    | AdamW     | SPP  | 88.3%      | 34.7           | 5.6         | SPP test       |

---

## 🧠 Future Work

- Combine multiple ablations (e.g., C3 + SGD + SPP) to test compound effects
- Evaluate on different classification datasets (e.g., CIFAR-100, TinyImageNet)
- Extend to detection/segmentation tasks

---
## 🔧 Setup

```bash
pip install -r requirements.txt

python split_dataset.py

python model.py

python predict.py path/to/image.jpg

Aerial_Landscapes/
├── Agriculture/
├── Beach/
├── City/
└── ...

runs/classify/train/weights/best.pt

