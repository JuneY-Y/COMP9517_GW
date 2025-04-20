import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import cv2
from models.swin_transformer_v2 import SwinTransformerV2
import os
from matplotlib import cm
# === 配置 ===
img_path = "datasets/train/Beach/009.jpg"
checkpoint_path = "outputs/swinv2_base_patch4_window8_256_with_pretrain-ft/swinv2_base_patch4_window8_256/default/ckpt_epoch_55.pth"
img_size = 256
num_classes = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 加载图片 ===
transform = T.Compose([
    T.Resize((img_size, img_size)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])
img_pil = Image.open(img_path).convert("RGB")
img_tensor = transform(img_pil).unsqueeze(0).to(device)

# === 构建模型 ===
model = SwinTransformerV2(
    img_size=img_size,
    window_size=8,
    num_classes=num_classes,
    embed_dim=128,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32]
).to(device)

ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
model.eval()

# === 注册最后的 patch feature hook ===
feature_maps = []
def hook_fn(module, input, output):
    feature_maps.append(output)

hook_handle = model.norm.register_forward_hook(hook_fn)

# === 前向传播以提取 feature ===
with torch.no_grad():
    _ = model(img_tensor)

# === 处理 feature map 成 attention mask ===
feat = feature_maps[0].squeeze(0)  # [N_patch, C]
feat_map = feat.mean(1)            # [N_patch]
patch_dim = int(np.sqrt(feat_map.shape[0]))
feat_map = feat_map.view(patch_dim, patch_dim).cpu().numpy()

feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min())
feat_map = cv2.resize(feat_map, img_pil.size)

# === 可视化叠图 ===
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_pil)
plt.axis("off")
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(img_pil)
plt.imshow(feat_map, cmap="jet", alpha=0.5)
plt.axis("off")
plt.title("Attention Overlay")
plt.tight_layout()
plt.show()
# ========= 可视化叠图 + 英文注释 + colorbar =========
fig, ax = plt.subplots(figsize=(6, 6))

ax.imshow(img_pil)
im = ax.imshow(feat_map, cmap="jet", alpha=0.5)
ax.axis("off")

# 添加 colorbar（legend）
cbar = fig.colorbar(im, ax=ax, shrink=0.8, orientation='vertical', pad=0.02)
cbar.set_label("Attention Strength", fontsize=12)

# 保存为文件
basename = os.path.basename(img_path)
output_path = os.path.join("output", f"attn_overlay_{os.path.splitext(basename)[0]}.png")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Saved overlay-only image to {output_path}")
