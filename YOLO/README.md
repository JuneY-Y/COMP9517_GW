YOLOv8 Aerial Image Classifier 🌍

Built by Jiaming Yang

This project uses YOLOv8’s classification mode to classify aerial landscape images (e.g., Beach, Forest, City).

⸻

🔧 Update

🧪 YOLOv8 Classification - Full Architectural Comparison

This repository extends ablation beyond SPP vs. SPPF and explores multiple architectural variants of YOLOv8 for aerial image classification.

We compare 20+ model variants with adjustments to:
	•	🧳 Feature blocks (C2f → C3)
	•	⚖️ Optimizers (AdamW vs. SGD)
	•	📊 Feature Fusion (SPPF vs. SPP)
	•	🛠️ Head structure (anchor-free vs. anchor-based)
	•	🤖 Attention modules (CBAM, SE)
	•	📈 Backbones (YOLOv8 vs. ViT)

⸻

💡 Design Motivation

We wanted to understand how specific architectural changes affect performance on a small, imbalanced, multi-class aerial dataset.
All models were trained with:
	•	⏱ 100 epochs
	•	📊 Batch size 256
	•	🌍 Image size 256x256
	•	☁️ Cloud environment (GPU accelerated)

Each script (e.g., model_attention.py, model_melt_v2_sgd.py) represents one experiment.

⸻

🔍 Ablation Highlights

1. C2f vs. C3 Blocks
	•	C3 improved accuracy and reduced parameter count compared to default C2f.

2. Optimizer: AdamW vs. SGD
	•	AdamW offered smoother convergence and better final accuracy.

3. SPPF vs. SPP
	•	SPP slightly improved accuracy but increased model size and training time.

4. Head: Anchor-Free vs. Anchor-Based
	•	Anchor-based heads slightly improved precision but increased inference latency.

⸻

📊 Results & Findings

⭐ Attention Model
	•	Achieved perfect accuracy (1.00) on critical classes like Lake, Residential
	•	Maintained stable generalization: 0.95 – 0.98 range on most classes

🔗 Fine-Tuned Model
	•	Showed minor gains, limited by lack of structural changes

📊 Best Model
	•	Balanced performance across all metrics (accuracy, size, FLOPs)
	•	Strong performance on difficult classes like Desert, Forest, Port

⚡ Challenging Classes
	•	Airport, River, and Highway remained hard to classify, likely due to low intra-class diversity and similar visual features

⸻

📆 Folder Structure

### 📁 Folder Structure Overview

| Folder/File          | Description                                                  |
|----------------------|--------------------------------------------------------------|
| `project/`           | Root directory of the project                                |
| ├── `model_*.py`     | 20+ model variants: C3, SPP, ViT, Attention, etc.            |
| ├── `modules/`       | Custom modules (e.g., SPP, CBAM, SE)                         |
| ├── `models/`        | YOLOv8 classification configs (`.yaml`)                      |
| ├── `datasets/`      | Aerial classification dataset: `train/`, `val/`, `test/`     |
| ├── `results/`       | Evaluation outputs: accuracy, F1-score, confusion matrix     |
| ├── `predict.py`     | Inference script for testing trained models                  |
| ├── `requirements.txt` | Python package dependencies                              |
| ├── `README.md`      | Project description (this file)                              |


⸻

✅ Example: Run Best Model

from ultralytics import YOLO
model = YOLO("models/yolov8n-cls-best.yaml")
model.train(data="datasets", epochs=100, imgsz=256, batch=256)



⸻

🚀 Conclusion

This study shows that small architectural tweaks can make a big difference — especially attention, feature fusion, and optimizer choice.
Even with a small, imbalanced dataset, we achieved strong generalization and over 95% accuracy using optimized YOLOv8 variants.
