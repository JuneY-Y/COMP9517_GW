from ultralytics import YOLO

# 替换为你的类别数量，比如 nc=15
model = YOLO("models/yolov8l-cls.yaml")

# 启动训练（你的数据集中 datasets/train 和 datasets/val 必须存在）
model.train(data="datasets", epochs=50, imgsz=224, batch=64)