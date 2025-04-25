"""
visualize_vit_patch_activation.py
---------------------------------
Overlay averaged ViT patch-token activations (blocks 1, 4, 7, 11) on top of
the original image.  Gives a rough sense of which regions each block attends to.
"""

import torch, timm, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2   # OpenCV for colourised overlay

# ---------------------------------------------------------------------
# 1.  Load ImageNet-pre-trained ViT-Base
# ---------------------------------------------------------------------
model = timm.create_model('vit_base_patch16_224', pretrained=True).eval()

# dict to store the features captured by forward-hooks
outs = {}

def make_hook(name):
    """Factory that returns a hook fn storing layer output in `outs`."""
    def hook(m, inp, out):
        outs[name] = out.detach()        # no grad, no tracking
    return hook

# attach hooks AFTER norm1 inside blocks 1, 4, 7, 11
layers = [1, 4, 7, 11]
for idx in layers:
    model.blocks[idx].norm1.register_forward_hook(make_hook(f'block{idx}'))

# ---------------------------------------------------------------------
# 2.  Pre-process a single image
# ---------------------------------------------------------------------
img_path = '/Users/yaogunzhishen/Desktop/未命名文件夹 7/datasets/test/Beach/009.jpg'
img = Image.open(img_path).convert('RGB')
orig_w, orig_h = img.size                             # for up-sampling later

tf = transforms.Compose([
    transforms.Resize((224, 224)),                    # ViT-Base input size
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])
x = tf(img).unsqueeze(0)                              # shape [1, 3, 224, 224]

# forward pass → hooks fill `outs`
with torch.no_grad():
    _ = model(x)

# ---------------------------------------------------------------------
# 3.  Build and show overlays
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for i, idx in enumerate(layers):
    # ---------------- Token features ----------------
    feat = outs[f'block{idx}']        # shape [1, 1+P, D] (CLS + patch tokens)
    patch_tokens = feat[0, 1:, :]     # drop CLS → [P, D]
    side = int(len(patch_tokens) ** 0.5)  # √P  (14 for 224/16)

    # reshape grid then average channels to obtain a scalar per patch
    patch_grid = patch_tokens.reshape(side, side, -1)   # [14, 14, D]
    heat = patch_grid.mean(-1).cpu().numpy()            # [14, 14]

    # ---------------- Post-process heat-map ----------
    heat_up = cv2.resize(heat, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    heat_smooth = gaussian_filter(heat_up, sigma=orig_h * 0.01)
    heat_norm = (heat_smooth - heat_smooth.min()) / (heat_smooth.ptp() + 1e-8)

    # colour-map to Jet → uint8 RGB
    cmap = plt.get_cmap('jet')
    heat_rgb = (cmap(heat_norm)[..., :3] * 255).astype(np.uint8)

    # ---------------- Blend with original ------------
    img_rgb = np.array(img)
    alpha = 0.5                                         # overlay opacity
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heat_rgb, alpha, 0)

    # ---------------- Plot ---------------------------
    ax = axes[i]
    ax.imshow(overlay)
    ax.set_title(f'Block {idx} overlay')
    ax.axis('off')

plt.tight_layout()
plt.show()