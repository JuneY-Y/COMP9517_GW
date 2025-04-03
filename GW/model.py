# ==============================================================================
# name: model.py
# aim: Train YOLOv8 image classification model
# ==============================================================================

from ultralytics import YOLO

model = YOLO("yolo8n-c3-cls.yaml")
# model.train(data="datasets", epochs=50, imgsz=224)

model.train(data='datasets', epochs=2, imgsz=128, batch=2 )