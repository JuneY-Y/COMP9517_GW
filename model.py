# ==============================================================================
# name: model.py
# aim: Train YOLOv8 image classification model
# ==============================================================================

from ultralytics import YOLO

# 加载 YOLOv8 分类模型
model = YOLO('yolov8n-cls.pt')

# 训练模型
model.train(data='datasets', epochs=50, imgsz=224, batch=10 )