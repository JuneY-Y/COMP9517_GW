# YOLOv8 Aerial Image Classifier 🌍
build by JiamingYang

This project uses **YOLOv8's classification mode** to classify aerial landscape images (e.g., Beach, Forest, City).

## 🔧 Update
# 🧪 YOLOv8 Classification - SPP vs. SPPF Ablation

This project tests what happens when we change just **one part** of the YOLOv8 classification model: the feature aggregation module.

---

## 💡 Design Idea

We designed a simple ablation experiment to see how the **feature aggregation** block affects the model.  
Everything stays the same — the **C2f backbone**, the **Classify head**, the **training settings**, and the **dataset** — except one thing:

🔄 We replaced the default **SPPF module** with a more traditional **SPP module**.

- **SPPF** is fast and efficient.
- **SPP** is older, but captures more information using different pooling sizes.

This experiment helps us understand:
> Does using a traditional method (SPP) give better accuracy or efficiency than the newer, faster SPPF?

---

## 🔍 What are SPP and SPPF?

### 🟩 SPP — Spatial Pyramid Pooling

- 📌 First used in YOLOv3 and YOLOv5
- 📦 Module name: `SPP`
- 🔧 Used in: YOLOv5's backbone and neck

#### 📐 How it works:

SPP stacks **multiple max-pooling layers** with different kernel sizes:
Input → MaxPool(5×5) → MaxPool(9×9) → MaxPool(13×13)
↘      ↘        ↘
concat → 1×1 Conv

- 📌 Captures **multi-scale features** (small, medium, large)  
- ❗ Needs **large kernels**, so it's slower and heavier  
- ✅ Can help with complex images by looking at different “zoom levels”

---

### ⚡ SPPF — Spatial Pyramid Pooling - Fast (default in YOLOv8)

- 🚀 Faster version of SPP
- 🔧 Used by default in YOLOv8
- 🔁 Uses 3 layers of **MaxPool(5×5)** in a row instead of 3 big separate ones

#### 🔍 Features:

- ✅ Very fast and efficient
- ✅ Uses less memory and computation
- 🔄 Still captures some multi-scale info — but in a simpler way

---

## 📊 Why this experiment matters

By switching just **SPPF → SPP**, we can compare:

| Metric              | What it tells us            |
|---------------------|-----------------------------|
| ✅ Accuracy          | Does SPP help the model predict better? |
| 🕒 Epoch Time        | Does SPP take longer to train?         |
| 🧠 Parameters (Params) | Does SPP make the model larger?       |
| 💾 Model Size        | Is the final `.pt` file bigger?         |

This helps us decide:  
**Is SPPF really better for small, fast models? Or is the old SPP still worth using?**

---

## 📁 Structure
project/
├── model.py                       # Training script
├── spp.py                         # Custom SPP module
├── models/
│   └── yolov8n-cls-spp.yaml       # Custom YOLOv8 model using SPP
├── datasets/
│   └── train/val/test             # Classification dataset

---

## ✅ Example Training Command

```python
from ultralytics import YOLO
from spp import SPP
import ultralytics.nn.modules
ultralytics.nn.modules.SPP = SPP

model = YOLO("models/yolov8n-cls-spp.yaml")
model.train(data="datasets", epochs=50, imgsz=224, batch=32)