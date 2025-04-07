# YOLOv8 Aerial Image Classifier 🌍
build by JiamingYang

This project uses **YOLOv8's classification mode** to classify aerial landscape images (e.g., Beach, Forest, City).

## 🔧 Setup

## Hyperparameters for YOLOv8 Classification by Jiaming

| Parameter        | Recommended Value | Notes (Easy English)                                             |
|------------------|-------------------|------------------------------------------------------------------|
| **imgsz**        | `224` or `256`    | Use `256` if you have a strong GPU, as it usually gives better results. |
| **batch**        | `64` or `128`     | Use `128` if you have lots of data; use `64` for smaller datasets.      |
| **epochs**       | `100` or `150`    | Usually, `100` is enough. If you have more data, use `150`.             |
| **lr0**          | `0.0001` or `0.0003` | AdamW optimizer often works better with smaller learning rates.      |
| **lrf**          | `0.01` or `0.005` | Final learning rate. Smaller values help training become more stable. |
| **weight_decay** | `0.001`           | AdamW optimizer works better with a slightly larger weight decay.    |
| **dropout**      | `0.1` or `0.2`    | Dropout helps the model avoid overfitting (learning too much detail). |
| **freeze**       | `10` or `15`      | Freeze the first 10~15 layers to help fine-tuning.                   |
| **optimizer**    | `'AdamW'`         | AdamW usually works better than SGD for classification tasks.         |
