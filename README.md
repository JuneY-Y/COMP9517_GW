# YOLOv8 Aerial Image Classifier 🌍
build by JiamingYang

This project uses **YOLOv8's classification mode** to classify aerial landscape images (e.g., Beach, Forest, City).

# 🔬 YOLOv8 Classification – C2f + SE + CBAM Attention Ablation

This experiment enhances the YOLOv8 classification backbone by inserting attention modules after each `C2f` block. We aim to evaluate the effect of adding both channel and spatial attention on classification performance, using a simple yet effective plug-in strategy.

---

## 🧠 What Are SE and CBAM?

### 🔹 SE (Squeeze-and-Excitation) Module

- Focuses on **channel-wise attention**.
- Learns to assign different importance to each feature channel.
- Mechanism:
  - Compress: Use global average pooling to get channel descriptors.
  - Excite: Apply two fully-connected layers to learn per-channel weights.
  - Scale: Re-weight the original features by multiplying learned weights.

> 📌 Helps the model focus on **what** is important.

### 🔸 CBAM (Convolutional Block Attention Module)

- Combines both **channel attention** and **spatial attention**.
- Applies attention in two sequential steps:
  1. **Channel Attention** – like SE
  2. **Spatial Attention** – applies 2D convolution to highlight "where" to focus.

> 📌 Helps the model focus on **where** and **what** at the same time.

---

## 💡 Design Strategy

We insert both `SE` and `CBAM` modules **after each C2f block** in the **backbone only**.

### ✅ Why after C2f?
- `C2f` is the core feature extractor in YOLOv8.
- Attention at this level directly influences low-to-mid-level feature representation.
- We avoid inserting in the **head** to keep the classification logic unchanged, making this a **clean ablation**.

### ✅ Why not everywhere?
- Adding attention globally (e.g., in the head) may mix feature enhancement with classification logic.
- Limiting it to the **backbone** keeps the test focused on improving feature representation.

---

## 📁 Modified Model Structure Overview
Conv → C2f → SE → CBAM → Conv → C2f → SE → CBAM → …

- All `C2f` blocks in the backbone are followed by both SE and CBAM.
- The `head` remains unchanged (using Classify).

---

## 🧪 Evaluation Goals

| Metric               | Description                                       |
|----------------------|---------------------------------------------------|
| 🔍 Top-1 / Top-5 Acc | Measure classification performance                |
| 🧠 Params (M)         | Measure total model parameter size                |
| 🕒 Epoch Time (s)     | Evaluate added training time from attention modules |
| 📦 Model Size (MB)   | Compare exported `.pt` weight sizes               |

---

## ✅ Expected Outcomes

| Model Variant          | Top-1 Acc | Params (M) | Epoch Time | Notes                     |
|------------------------|-----------|-------------|------------|----------------------------|
| YOLOv8n-cls (baseline) | 88.2%     | 5.1M        | 32.5s      | No attention               |
| + SE only              | 88.7%     | ↑ ~5.3M     | ~33.1s     | Channel-wise attention     |
| + CBAM only            | 89.1%     | ↑ ~5.5M     | ~33.8s     | Channel + spatial          |
| + SE + CBAM            | **89.4%** | ↑ ~5.6M     | ~34.1s     | Combined attention modules |

---

## 🧠 Summary

> This ablation isolates the effect of inserting SE and CBAM attention modules into the backbone of YOLOv8. By modifying only the feature extraction path (C2f outputs), we can evaluate whether channel-wise and spatial attention significantly improve classification performance, without changing the prediction logic or model head.
>
> ## idea follow paper by Jiaming

---
