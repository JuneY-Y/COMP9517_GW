from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import os


def train_model_basic():
    # ✅ 1. 加载预训练分类模型（不做任何 freeze 或结构修改）
    model = YOLO("yolov8n-cls.pt")

    # ✅ 2. 开始训练，保持默认结构与优化器配置
    model.train(
        data="datasets",
        epochs=50,
        imgsz=256,
        batch=128,
        project="runs/classify",
        name="basic_yolov8",
        patience=10  # ✅ early stopping
    )

    # ✅ 3. 模型评估
    metrics = model.val()
    print(f"\n📊 Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"📊 Top-5 Accuracy: {metrics.top5:.4f}")

    # ✅ 4. 训练日志处理
    results_dir = os.path.join("runs", "classify", "basic_yolov8")
    results_path = os.path.join(results_dir, "results.csv")
    best_model_path = os.path.join(results_dir, "weights", "best.pt")

    if os.path.exists(results_path):
        df = pd.read_csv(results_path)

        # ✅ 备份结果 CSV
        df.to_csv(os.path.join(results_dir, "results_saved.csv"), index=False)

        # ✅ 绘图（loss、accuracy、f1）
        plt.figure(figsize=(15, 4))

        # Loss 曲线
        plt.subplot(1, 3, 1)
        plt.plot(df["epoch"], df["train/loss"], label="Train Loss")
        plt.plot(df["epoch"], df["val/loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()

        # Accuracy 曲线
        plt.subplot(1, 3, 2)
        plt.plot(df["epoch"], df["metrics/accuracy_top1"], label="Top-1 Acc")
        plt.plot(df["epoch"], df["metrics/accuracy_top5"], label="Top-5 Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()

        # F1 曲线
        plt.subplot(1, 3, 3)
        if "metrics/f1" in df.columns:
            plt.plot(df["epoch"], df["metrics/f1"], label="F1 Score", color="green")
            plt.xlabel("Epoch")
            plt.ylabel("F1 Score")
            plt.title("F1-score Curve")
            plt.legend()
        else:
            print("⚠️ 日志中未检测到 F1-score")

        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "curve.png"))
        print("📈 训练曲线已保存：curve.png")

        # 最佳模型提示
        if os.path.exists(best_model_path):
            print(f"💾 最佳模型已保存：{best_model_path}")
        else:
            print("⚠️ 未检测到 best.pt，可能未触发 early stopping")

    else:
        print("❌ 未找到 results.csv，无法生成日志图")


if __name__ == "__main__":
    train_model_basic()