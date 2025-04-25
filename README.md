📊 Project Overview: Aerial Image Classification Benchmark

This repository presents a comprehensive benchmark study on aerial landscape classification across five model families: traditional ML, CNNs (YOLOv8, ResNet), and Transformers (ViT, Swin V2). The goal is to evaluate accuracy, robustness, and long-tail performance under unified settings.

⸻

🏋️ Total Workload Summary

We implemented, trained, and compared 20+ model variants, evaluated over 3 benchmark scenarios:
	•	Baseline Testing on balanced data
	•	Robustness Evaluation under noise, distortion, and compression
	•	Long-Tail Testing under severe class imbalance (IR=50)

✅ Each experiment includes ablations, accuracy, macro-F1, model size, and FLOPs.

⸻

💡 Main Experiments Breakdown

1. Traditional Machine Learning (10 classifiers × 6 PCA settings)
	•	Extracted Color-LBP + Color-SIFT
	•	Conducted 414 hyperparameter-PCA pipelines
	•	Benchmarked with perturbation and long-tail variants

2. YOLOv8 Variants (10+ ablation studies)
	•	C2f vs. C3, SPPF vs. SPP, Anchor-Free vs. Anchor-Based
	•	SE/CBAM Attention, ViT-Backbone hybrid
	•	KL Consistency, MixUp, Mosaic, LMA, etc.

3. ResNet Backbone + Classifier Swaps
	•	Compared fc, SVM, MLP, KNN, ProtoNet
	•	Re-sampling for imbalance
	•	Perturbation-aware retraining

4. ViT (Base)
	•	Position Encoding and Transformer Layer Ablation
	•	Two-Stage Training
	•	LMA for perturbation robustness

5. Swin Transformer V2
	•	Fine-tuned with 7 augmentation strategies
	•	Tested StepLR, Full Aug, and Tiny variant
	•	Ablated depth, shift-windows, window size
	•	Best long-tail & robustness performance

---

## 💡 Main Experiments Breakdown
For detailed configurations of each model, please refer to the corresponding model folder.
### [📁 Traditional Machine Learning](./Machine%20Learning/)
- 10 classifiers × 6 PCA settings
- Extracted Color-LBP + Color-SIFT features
- Conducted 414 hyperparameter-PCA pipelines
- Benchmarked with perturbation and long-tail variants

### [📁 YOLOv8 Variants](./YOLO/)
- C2f vs. C3, SPPF vs. SPP, Anchor-Free vs. Anchor-Based
- SE/CBAM Attention, ViT-Backbone hybrid
- KL Consistency, MixUp, Mosaic, LMA, etc.

### [📁 ResNet Backbone + Classifier Swaps](./ResNet/)
- Compared fc, SVM, MLP, KNN, ProtoNet
- Re-sampling for imbalance
- Perturbation-aware retraining

### [📁 ViT (Base)](./VIT%20model/)
- Position Encoding and Transformer Layer Ablation
- Two-Stage Training
- LMA for perturbation robustness

### [📁 Swin Transformer V2](./Swin-Transformer/)
- Fine-tuned with 7 augmentation strategies
- Tested StepLR, Full Aug, and Tiny variant
- Ablated depth, shift-windows, window size
- Best long-tail & robustness performance

---


## 🧵 Key Outcomes

| Model                        | Top-1 Acc | Macro-F1 | Robust ΔL1–L3 | Tail-Aware |
|-----------------------------|-----------|----------|----------------|-------------|
| SwinV2 + Dynamic ASL        | 99.1%     | 0.991    | ↓ 6.6%         | ✅ Best      |
| ViT + Two-Stage + LMA       | 97.6%     | 0.976    | ↓ 7.1%         | ✅           |
| YOLOv8n + CBAM              | 98.0%     | 0.980    | ↓ 26.6%        | ⭐ Moderate  |
| ResNet-50 + SVM             | 96.1%     | 0.961    | ↓ 28.8%        | ✅           |
| Traditional Voting          | 86.9%     | 0.869    | ↓ 12.9%        | ⭐ Basic     |

---




### 📊 Long-Tail Performance After Mitigation

| Model & Strategy                        | Overall Top-1 (%) | Macro-F1 |
|----------------------------------------|--------------------|----------|
| Balanced Random Forest                 | 73.50              | 0.736    |
| ResNet-50 + SVM + CA-RS                | 90.83              | 0.910    |
| YOLOv8 + KL consistency                | 70.26              | 0.702    |
| ViT-Base + Two-Stage                   | 94.83              | 0.946    |
| **Swin V2 + Dynamic ASL**              | **98.67**          | **0.987** |

---

### 🧪 Robustness After Perturbation-Resilience Optimization

| Model & Strategy                     | Level 1 (light) | Level 2 (medium) | Level 3 (heavy) | Δ (L1–L3) |
|-------------------------------------|-----------------|------------------|------------------|-----------|
| Voting + level‑matched retrain      | 84.00           | 78.17            | 71.06            | 12.94     |
| ResNet‑50 + SVM + perturb           | 95.08           | 87.71            | 66.24            | 28.84     |
| YOLOv8 + KL consistency             | 96.83           | 89.47            | 70.26            | 26.57     |
| **Swin V2 + LMA**                   | **97.69**       | **97.06**        | **91.13**        | **6.56**  |
| ViT‑Base + LMA                      | 96.72           | 95.44            | 89.61            | 7.11      |

---
📅 Timeline Overview
	•	✍️ 20+ models trained (CNNs, ViTs, Swin, ML classifiers)
	•	⚖️ 100+ hours of tuning (aug, loss, optimizer, layers)
	•	📊 ~50 ablation experiments across 3 benchmarks
	•	🪡 6 key robustness and long-tail strategies tested

⸻

🚀 Conclusion

This benchmark provides a deep and practical comparison of models, methods, and mitigation strategies in aerial image classification. We hope this unified repo aids future research in robust, fair, and efficient remote sensing AI.
