from ultralytics import YOLO
import multiprocessing

def train_and_evaluate():
    # ✅ 1. 加载预训练模型（Fine-tuning）
    model = YOLO("yolov8n-cls.pt")

    # ✅ 2. 推荐的超参数组合
    model.train(
        data="datasets",
        epochs=100,
        imgsz=256,
        batch=128,
        lr0=0.0001,
        lrf=0.005,
        weight_decay=0.001,
        dropout=0.2,
        project="runs/classify",
        name="finetune_best",
        pretrained=True,
        freeze=10,
        optimizer='AdamW'
    )

    # ✅ 3. 自动评估
    metrics = model.val()

    # ✅ 4. 打印 top1 / top5
    print(f"\n📊 Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"📊 Top-5 Accuracy: {metrics.top5:.4f}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_and_evaluate()

# from ultralytics import YOLO
# import multiprocessing
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import numpy as np
# import os
#
#
# def train_and_evaluate():
#     # ✅ 1. 加载预训练模型（Fine-tuning）
#     model = YOLO("yolov8n-cls.pt")
#
#     # ✅ 2. 训练配置
#     model.train(
#         data="datasets",  # 注意：这个是文件夹路径，不是 .yaml！
#         epochs=100,
#         imgsz=256,
#         batch=128,
#         lr0=0.0001,
#         lrf=0.01,
#         weight_decay=0.0005,  # ← 权重衰减
#         dropout=0,
#         project="runs/classify",
#         name="finetune_cls",
#         pretrained=True,  # ← 明确说明使用预训练
#         freeze=0,  # ← 可设置为冻结前 N 层（如 10）
#         optimizer='AdamW'
#     )
#
#     # ✅ 3. 自动评估
#     metrics = model.val()
#
#     # ✅ 4. 打印 top1 / top5
#     print(f"\n📊 Top-1 Accuracy: {metrics.top1:.4f}")
#     print(f"📊 Top-5 Accuracy: {metrics.top5:.4f}")
#
#
#
# if __name__ == '__main__':
#     multiprocessing.freeze_support()  # Windows 需要
#     train_and_evaluate()
