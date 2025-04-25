# ==============================================================================
# name: model_spp.py
# aim: Train YOLOv8 classification model with SPP instead of SPPF
# ==============================================================================

from ultralytics import YOLO
from spp import SPP

# Register SPP module with Ultralytics
import ultralytics.nn.modules
ultralytics.nn.modules.SPP = SPP

# Load model with SPP instead of SPPF
model = YOLO("models/yolov8n-cls-spp.yaml")

# Start training
model.train(data="datasets", epochs=2, imgsz=128, batch=2)