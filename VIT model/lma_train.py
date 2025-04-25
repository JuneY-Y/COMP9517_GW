"""
level_matched_aug_vit.py
------------------------
• Three-level test-time perturbation recipe (geo / colour / noise).
• On-the-fly data augmentation during training.
• Consistency regularisation: KL(p_aug ‖ p_orig).
• Early-stopping on validation accuracy.
"""

import os, time, random, numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

# ---------------------------------------------------------------------#
# 1.  Augmentation config (3 severity levels)                          #
# ---------------------------------------------------------------------#
LEVEL_CONFIG = {
    1: dict(geo_prob=0.3, angle_range=15,
            color_prob=0.3, bright_range=(0.9,1.1), contrast_range=(0.9,1.1),
            saturation_range=(0.9,1.1), hue=0.05,
            noise_prob=0.3, noise_std=5),
    2: dict(geo_prob=0.5, angle_range=30,
            color_prob=0.5, bright_range=(0.8,1.2), contrast_range=(0.8,1.2),
            saturation_range=(0.8,1.2), hue=0.10,
            noise_prob=0.5, noise_std=15),
    3: dict(geo_prob=0.8, angle_range=45,
            color_prob=0.8, bright_range=(0.7,1.3), contrast_range=(0.7,1.3),
            saturation_range=(0.7,1.3), hue=0.20,
            noise_prob=0.8, noise_std=25),
}

# ---------------------------------------------------------------------#
# 2.  Helper functions                                                 #
# ---------------------------------------------------------------------#
def add_noise(img: Image.Image, std: float) -> Image.Image:
    """Gaussian noise in the RGB space (per-pixel)."""
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def augment_image(img: Image.Image, level: int = 2) -> Image.Image:
    """Apply randomised geometric + colour + noise transforms."""
    cfg = LEVEL_CONFIG[level]
    out = img.copy()

    # -- geometric ----------------------------------------------------
    if random.random() < cfg["geo_prob"]:
        angle = random.uniform(-cfg["angle_range"], cfg["angle_range"])
        out = out.rotate(angle, expand=True)
    if random.random() < cfg["geo_prob"]:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < cfg["geo_prob"]:
        out = out.transpose(Image.FLIP_TOP_BOTTOM)

    # -- colour -------------------------------------------------------
    if random.random() < cfg["color_prob"]:
        jitter = transforms.ColorJitter(brightness=cfg["bright_range"],
                                        contrast=cfg["contrast_range"],
                                        saturation=cfg["saturation_range"],
                                        hue=cfg["hue"])
        out = jitter(out)

    # -- Gaussian sensor noise ---------------------------------------
    if random.random() < cfg["noise_prob"]:
        out = add_noise(out, std=cfg["noise_std"])

    return out

# ---------------------------------------------------------------------#
# 3.  Base transform (resize-crop-norm)                                #
# ---------------------------------------------------------------------#
base_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ---------------------------------------------------------------------#
# 4.  Dataset & dataloaders                                            #
# ---------------------------------------------------------------------#
train_ds = datasets.ImageFolder("datasets/train", transform=base_tf)
val_ds   = datasets.ImageFolder("datasets/test",  transform=base_tf)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)

# ---------------------------------------------------------------------#
# 5.  ViT-Base model (ImageNet-21k pre-train ckpt)                     #
# ---------------------------------------------------------------------#
num_classes = len(train_ds.classes)
model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=False,
    checkpoint_path="B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    num_classes=num_classes,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------------------------------------------------#
# 6.  Optimiser / scheduler                                            #
# ---------------------------------------------------------------------#
criterion  = nn.CrossEntropyLoss()
optimizer  = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ---------------------------------------------------------------------#
# 7.  Training - Level-Matched Aug + Consistency                       #
# ---------------------------------------------------------------------#
λ_cons        = 0.5          # weight for KL consistency term
num_epochs    = 50
patience      = 5
best_val_acc  = 0.0
patience_cnt  = 0

for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # -- on-the-fly augmentation ---------------------------------
        pil_batch   = [ transforms.ToPILImage()(x.cpu()) for x in imgs ]
        levels      = [ random.choice((1,2,3))           for _ in pil_batch ]
        aug_batch   = [ augment_image(p, l)              for p, l in zip(pil_batch, levels) ]
        imgs_aug    = torch.stack([ base_tf(p) for p in aug_batch ]).to(device)

        # -- forward & losses ----------------------------------------
        logits_orig = model(imgs)        # clean view
        logits_aug  = model(imgs_aug)    # perturbed view

        loss_ce   = criterion(logits_aug, labels)
        p_orig    = F.softmax(logits_orig, dim=1)
        p_aug     = F.softmax(logits_aug,  dim=1)
        loss_cons = F.kl_div(p_aug.log(), p_orig, reduction="batchmean")

        loss = loss_ce + λ_cons * loss_cons

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    # -- epoch stats --------------------------------------------------
    train_loss = running_loss / len(train_ds)
    scheduler.step()

    # -- validation ---------------------------------------------------
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    val_acc = correct / total
    sec = time.time() - t0

    print(f"[{epoch:02d}/{num_epochs}] "
          f"loss={train_loss:.4f}  val_acc={val_acc:.4f}  time={sec:.1f}s")

    # -- early-stopping ------------------------------------------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_cnt = 0
        torch.save(model.state_dict(), "vit_lma_best.pth")
        print("  ↳ new best saved.")
    else:
        patience_cnt += 1
        if patience_cnt >= patience:
            print("Early stopping triggered.")
            break

print(f"Training complete | best val_acc={best_val_acc:.4f}")