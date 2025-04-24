import torch, timm, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2  


model = timm.create_model('vit_base_patch16_224', pretrained=True).eval()
outs = {}

def make_hook(name):
    def hook(m, inp, out):
        outs[name] = out.detach()
    return hook


layers = [1, 4, 7, 11]
for idx in layers:
    model.blocks[idx].norm1.register_forward_hook(make_hook(f'block{idx}'))


img = Image.open('/Users/yaogunzhishen/Desktop/未命名文件夹 7/datasets/test/Beach/009.jpg').convert('RGB')
orig_w, orig_h = img.size
tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
x = tf(img).unsqueeze(0)  


with torch.no_grad():
    _ = model(x)



fig, axes = plt.subplots(2, 2, figsize=(12,12))
axes = axes.flatten()

for i, idx in enumerate(layers):
    feat = outs[f'block{idx}']           # [1, 1+P, D]
    P = feat.shape[1] - 1
    tok = feat[0,1:,:]                   # [P, D]
    side = int(P**0.5)
    tok = tok.reshape(side, side, -1)    # [side, side, D]
    heat = tok.mean(-1).cpu().numpy()    # [side, side]

    heat_up = cv2.resize(heat, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    heat_smooth = gaussian_filter(heat_up, sigma=orig_h*0.01)
    heat_smooth = (heat_smooth - heat_smooth.min()) / (heat_smooth.max() - heat_smooth.min() + 1e-8)


    cmap = plt.get_cmap('jet')
    heat_color = cmap(heat_smooth)[:, :, :3]
    heat_color = (heat_color * 255).astype(np.uint8)


    img_np = np.array(img)


    alpha = 0.5
    overlay = cv2.addWeighted(img_np, 1-alpha, heat_color, alpha, 0)


    ax = axes[i]
    ax.imshow(overlay)
    ax.set_title(f'Block {idx} Overlay')
    ax.axis('off')

plt.tight_layout()
plt.show()