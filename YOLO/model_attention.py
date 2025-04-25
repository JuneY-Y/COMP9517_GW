import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks  # 导入任务模块，用于修改其命名空间


# ========== SE Block ==========
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ========== CBAM Block ==========
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        x = x * self.sigmoid_channel(avg_out + max_out)

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.sigmoid_spatial(self.conv_spatial(spatial))
        return x * spatial


# ========== SE + CBAM Combined ==========
class SE_CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.se = SEBlock(channels)
        self.cbam = CBAMBlock(channels)

    def forward(self, x):
        return self.cbam(self.se(x))


# 将自定义模块注册到 ultralytics.nn.tasks 模块的命名空间中
tasks.__dict__["SE_CBAM"] = SE_CBAM

if __name__ == "__main__":
    # 加载 YAML 模型配置文件（请确保路径正确）
    model = YOLO(r"F:\study\9517\COMP9517_GW-Jiaming_melt\models\yolo8n-c3-cls-attn.yaml")

    # 开始训练
    model.train(data=r"F:\study\9517\COMP9517_GW-Jiaming_melt\datasets", epochs=100, imgsz=256, batch=64)