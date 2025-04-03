# YOLOv8 Aerial Image Classifier 🌍
build by JiamingYang

This project uses **YOLOv8's classification mode** to classify aerial landscape images (e.g., Beach, Forest, City).

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


