"""
train_vit_basic.py
------------------
Minimal training script for ViT-Base on an ImageFolder dataset.

Key features
============
*   Random-crop + flip augmentation for the training split
*   Center-crop for the validation split
*   ImageNet-21k checkpoint for weight initialisation
*   AdamW optimiser + StepLR schedule
*   Early-stopping on validation accuracy
*   Best weights stored in ``best_model.pth``
"""

import time, torch, timm
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ------------------------------------------------------------------
# 1.  Data transforms
# ------------------------------------------------------------------
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),      # keep AR, then …
    transforms.RandomCrop(224),         # … random 224×224 crop
    transforms.RandomHorizontalFlip(),  # left–right flip, p=0.5
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),         # deterministic crop
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ------------------------------------------------------------------
# 2.  ImageFolder datasets & loaders
# ------------------------------------------------------------------
train_ds = datasets.ImageFolder('datasets/train', transform=train_tf)
val_ds   = datasets.ImageFolder('datasets/test',  transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)

# ------------------------------------------------------------------
# 3.  Build ViT-Base with ImageNet-21k weights (head ignored)
# ------------------------------------------------------------------
num_classes = len(train_ds.classes)

model = timm.create_model(
    'vit_base_patch16_224',
    pretrained=False,
    checkpoint_path=(
        'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--'
        'imagenet2012-steps_20k-lr_0.01-res_224.npz'
    ),
    num_classes=num_classes,            # replaces classification head
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ------------------------------------------------------------------
# 4.  Loss, optimiser, LR scheduler
# ------------------------------------------------------------------
criterion  = nn.CrossEntropyLoss()
optimizer  = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ------------------------------------------------------------------
# 5.  Training hyper-params / early-stop setup
# ------------------------------------------------------------------
NUM_EPOCHS       = 50
PATIENCE         = 5                    # epochs to wait before early stop
best_val_acc     = 0.0
patience_counter = 0
SAVE_PATH        = 'best_model.pth'

# ------------------------------------------------------------------
# 6.  Training loop
# ------------------------------------------------------------------
overall_start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0

    # ---------- forward / backward on training data ----------
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    train_loss = running_loss / len(train_ds)
    scheduler.step()                     # StepLR update

    # ---------- validation pass ----------
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    val_acc = correct / total

    # ---------- logging ----------
    dt = time.time() - epoch_start
    print(f"[{epoch:02d}/{NUM_EPOCHS}] "
          f"loss={train_loss:.4f}  val_acc={val_acc:.4f}  time={dt:.1f}s")

    # ---------- checkpoint / early-stop logic ----------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print("  ↳ New best model saved.")
    else:
        patience_counter += 1
        print(f"  ↳ No improvement for {patience_counter} epoch(s).")

    if patience_counter >= PATIENCE:
        print("Early stopping triggered.")
        break

# ------------------------------------------------------------------
# 7.  Final stats
# ------------------------------------------------------------------
print(f"Training finished | best val_acc={best_val_acc:.4f} | "
      f"total time={ (time.time()-overall_start)/60:.1f} min")