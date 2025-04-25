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

⸻

🧵 Key Outcomes

Model	Top-1 Acc	Macro-F1	Robust ΔL1-L3	Tail-Aware
SwinV2 + Dynamic ASL	99.1%	0.991	↓ 6.6%	✅ Best
ViT + Two-Stage + LMA	97.6%	0.976	↓ 7.1%	✅
YOLOv8n + CBAM	98.0%	0.980	↓ 26.6%	⭐ Moderate
ResNet-50 + SVM	96.1%	0.961	↓ 28.8%	✅
Traditional Voting	86.9%	0.869	↓ 12.9%	⭐ Basic



⸻

📅 Timeline Overview
	•	✍️ 20+ models trained (CNNs, ViTs, Swin, ML classifiers)
	•	⚖️ 100+ hours of tuning (aug, loss, optimizer, layers)
	•	📊 ~50 ablation experiments across 3 benchmarks
	•	🪡 6 key robustness and long-tail strategies tested

⸻

🚀 Conclusion

This benchmark provides a deep and practical comparison of models, methods, and mitigation strategies in aerial image classification. We hope this unified repo aids future research in robust, fair, and efficient remote sensing AI.
