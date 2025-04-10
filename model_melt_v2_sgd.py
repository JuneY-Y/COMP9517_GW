from ultralytics import YOLO

# 加载你定义的模型结构（保持为 yolov8n-cls.yaml）
model = YOLO('yolov8n-cls.yaml')

# 开始训练，优化器为 SGD
model.train(data='datasets',
            epochs=50,
            imgsz=224,
            optimizer='SGD',       # ✅ 替换 AdamW 为 SGD
            lr0=0.01,              # 初始学习率，建议用 SGD 时稍高一些
            momentum=0.937,        # SGD 的动量项
            weight_decay=0.0005)   # 权重衰减