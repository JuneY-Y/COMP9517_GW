# from ultralytics import YOLO
#
# model = YOLO("yolov8n-cls.pt")
#
# model.train(
#     data="datasets",
#     epochs=100,
#     imgsz=256,
#     batch=128,
#     degrees=90,         # 航拍图像旋转90度合理
#     translate=0.1,      # 小幅度平移
#     scale=0.5,          # 缩放 ±50%
#     fliplr=0.5,         # 左右翻转概率50%
#     flipud=0.5,         # 垂直翻转概率50%（航拍适合）
#     mosaic=1.0,         # Mosaic增强
#     mixup=0.2,          # Mixup增强
#     hsv_h=0.015,        # 颜色变化（轻微）
#     hsv_s=0.7,          # 饱和度变化
#     hsv_v=0.4,          # 亮度变化
#     blur=0.1,           # 轻微模糊 is not be promitted
#     cutout=0.05         # 轻微遮挡
# )
import multiprocessing

from ultralytics import YOLO


def train_and_evaluate():
    model = YOLO("yolov8n-cls.pt")
    model.train(
        data="datasets",
        epochs=100,
        imgsz=256,
        batch=128,
        degrees=90,         # 旋转 ±90°
        translate=0.1,      # 平移 10%
        scale=0.5,          # 缩放 ±50%
        fliplr=0.5,         # 左右翻转概率
        flipud=0.5,         # 上下翻转概率（航拍图像适合）
        mosaic=1.0,         # Mosaic增强
        mixup=0.2,          # Mixup增强
        hsv_h=0.015,        # 色调变化
        hsv_s=0.7,          # 饱和度变化
        hsv_v=0.4           # 亮度变化
    )

  # ✅ 3. 自动评估
    metrics = model.val()

    # ✅ 4. 打印 top1 / top5
    print(f"\n📊 Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"📊 Top-5 Accuracy: {metrics.top5:.4f}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_and_evaluate()