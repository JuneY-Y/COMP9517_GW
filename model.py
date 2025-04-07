from ultralytics import YOLO
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import os

def train_and_evaluate():
    # ✅ 1. 加载预训练模型（Fine-tuning）
    model = YOLO("yolov8n-cls.pt")

    # ✅ 2. 设置训练参数，加入 Early Stopping 配置
    model.train(
        data="datasets",
        epochs=50,
        imgsz=256,
        batch=128,
        lr0=0.0001,
        weight_decay=0.001,
        dropout=0.2,
        project="runs/classify",
        name="finetune_best2",
        pretrained=True,
        # freeze=0,
        optimizer="AdamW",
        patience=10  # ✅ 提前终止训练（early stopping），10个 epoch 没提升就停
    )

    # ✅ 3. 自动评估
    metrics = model.val()

    # ✅ 4. 打印 top1 / top5
    print(f"\n📊 Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"📊 Top-5 Accuracy: {metrics.top5:.4f}")

    # ✅ 5. 读取训练日志（Ultralytics 自动保存 results.csv）
    results_dir = os.path.join("runs", "classify", "finetune_best2")
    results_path = os.path.join(results_dir, "results.csv")
    best_model_path = os.path.join(results_dir, "weights", "best.pt")

    if os.path.exists(results_path):
        df = pd.read_csv(results_path)

        # ✅ 6. 保存为副本 results_saved.csv（防止被覆盖）
        df.to_csv(os.path.join(results_dir, "results_saved.csv"), index=False)

        # ✅ 7. 绘制训练曲线（Loss、Top1、F1）
        plt.figure(figsize=(15, 4))

        # ✅ Loss 曲线
        plt.subplot(1, 3, 1)
        plt.plot(df['epoch'], df['train/loss'], label='Train Loss')
        plt.plot(df['epoch'], df['val/loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

        # ✅ Top-1 Accuracy 曲线
        plt.subplot(1, 3, 2)
        plt.plot(df['epoch'], df['metrics/accuracy_top1'], label='Top-1 Accuracy')
        plt.plot(df['epoch'], df['metrics/accuracy_top5'], label='Top-5 Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()

        # ✅ F1-score 曲线
        if 'metrics/f1' in df.columns:
            plt.subplot(1, 3, 3)
            plt.plot(df['epoch'], df['metrics/f1'], label='F1-score', color='green')
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.title('F1-score Curve')
            plt.legend()
        else:
            print("⚠️ 当前日志中未检测到 'metrics/f1'，可能该模型不支持")

        # ✅ 8. 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "curve.png"))
        print("📈 训练曲线保存完成 ✅")

        # ✅ 9. 输出 best 模型路径
        if os.path.exists(best_model_path):
            print(f"\n💾 最佳模型已保存：{best_model_path}")
        else:
            print("⚠️ 未找到 best.pt，可能训练提前中断或未保存")

    else:
        print("❌ 未找到 results.csv，无法绘图。")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_and_evaluate()

# from ultralytics import YOLO
# import multiprocessing
#
# def train_and_evaluate():
#     # ✅ 1. 加载预训练模型（Fine-tuning）
#     model = YOLO("yolov8n-cls.pt")
#
#     # ✅ 2. 推荐的超参数组合
#     model.train(
#         data="datasets",
#         epochs=100,
#         imgsz=256,
#         batch=128,
#         lr0=0.0001,
#         weight_decay=0.001,
#         dropout=0.2,
#         project="runs/classify",
#         name="finetune_best2",
#         pretrained=True,
#         freeze=10,
#         optimizer="AdamW"  # ✅ 保留
#         # 不要手动加 momentum 参数
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
# if __name__ == '__main__':
#     multiprocessing.freeze_support()
#     train_and_evaluate()
#
# # from ultralytics import YOLO
# # import multiprocessing
# # import matplotlib.pyplot as plt
# # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# # import numpy as np
# # import os
# #
# #
# # def train_and_evaluate():
# #     # ✅ 1. 加载预训练模型（Fine-tuning）
# #     model = YOLO("yolov8n-cls.pt")
# #
# #     # ✅ 2. 训练配置
# #     model.train(
# #         data="datasets",  # 注意：这个是文件夹路径，不是 .yaml！
# #         epochs=100,
# #         imgsz=256,
# #         batch=128,
# #         lr0=0.0001,
# #         lrf=0.01,
# #         weight_decay=0.0005,  # ← 权重衰减
# #         dropout=0,
# #         project="runs/classify",
# #         name="finetune_cls",
# #         pretrained=True,  # ← 明确说明使用预训练
# #         freeze=0,  # ← 可设置为冻结前 N 层（如 10）
# #         optimizer='AdamW'
# #     )
# #
# #     # ✅ 3. 自动评估
# #     metrics = model.val()
# #
# #     # ✅ 4. 打印 top1 / top5
# #     print(f"\n📊 Top-1 Accuracy: {metrics.top1:.4f}")
# #     print(f"📊 Top-5 Accuracy: {metrics.top5:.4f}")
# #
# #
# #
# # if __name__ == '__main__':
# #     multiprocessing.freeze_support()  # Windows 需要
# #     train_and_evaluate()
