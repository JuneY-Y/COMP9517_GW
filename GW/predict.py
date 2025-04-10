# predict.py
from ultralytics import YOLO
import sys

model = YOLO("runs/classify/train/weights/best.pt")

image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
results = model(image_path)

top1 = results[0].probs.top1
prob = results[0].probs.top1conf.item()
print(f"✅ Predicted class: {results[0].names[top1]} (confidence: {prob:.2f})")