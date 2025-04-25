from ultralytics import YOLO

# 1. 加载 Tiny 模型结构（不使用预训练）
model = YOLO("models/yolov8n-cls.yaml")  # 确保这个 .yaml 文件存在

# 2. 开始训练分类模型
model.train(
    data="datasets",          # 包含 train/val 子目录
    epochs=50,
    imgsz=224,
    batch=64,
    project="runs/classify",
    name="yolov8n",
    patience=10
)