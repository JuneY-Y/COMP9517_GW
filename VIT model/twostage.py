#!/usr/bin/env python
# vit_two_stage_longtail.py
# ---------------------------------------------------------
# ViT-Base two-stage training on a long-tailed dataset
#   Stage 1: train on the full data
#   Stage 2: freeze the backbone and fine-tune the classifier
#            head only on tail classes
# ---------------------------------------------------------

import os, time, random, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm

# ---------- 0. Global settings ----------
ROOT          = 'longtail/longtail'   # dataset root (train/  test/)
BATCH         = 32
NUM_WORKERS   = 4
TAIL_THRESH   = 100                   # <100 samples → tail class
STAGE1_EPOCHS = 50
STAGE2_EPOCHS = 20
PATIENCE      = 5                     # early-stopping patience
SEED          = 42

# ---------- 1. Fix random seeds (reproducibility) ----------
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Select GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on', device)

# ---------- 2. Data augmentation ----------
train_tf = transforms.Compose([
    transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
val_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(root=os.path.join(ROOT, 'train'), transform=train_tf)
val_ds   = datasets.ImageFolder(root=os.path.join(ROOT, 'test'),  transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

num_classes = len(train_ds.classes)
print(f'Dataset: {num_classes} classes | {len(train_ds)} train | {len(val_ds)} val')

# ---------- 3. Build the tail-class DataLoader ----------
freq         = np.bincount([y for _, y in train_ds.imgs], minlength=num_classes)
tail_classes = np.where(freq < TAIL_THRESH)[0].tolist()
print(f'Tail classes (<{TAIL_THRESH} images): {tail_classes}')

tail_idx   = [i for i, (_, y) in enumerate(train_ds.imgs) if y in tail_classes]
tail_ds    = Subset(train_ds, tail_idx)
tail_loader = DataLoader(tail_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS)

# ---------- 4. Stage 1 — train on the full data ----------
model_name = 'vit_base_patch16_224'
model = timm.create_model(model_name,
                          pretrained=True,          # ImageNet-21k weights
                          num_classes=num_classes).to(device)

criterion  = nn.CrossEntropyLoss()
optimizer  = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

best_acc, wait = 0.0, 0
ckpt_stage1 = 'vit_longtail_stage1.pth'
print('\n========= Stage 1 (full data) =========')

for epoch in range(1, STAGE1_EPOCHS + 1):
    t0 = time.time()
    model.train()
    running_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)

    scheduler.step()
    train_loss = running_loss / len(train_ds)

    # ----- validation -----
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            total   += y.size(0)
            correct += (pred == y).sum().item()
    val_acc = correct / total
    print(f'[Epoch {epoch:02d}/{STAGE1_EPOCHS}] '
          f'loss={train_loss:.4f}  val_acc={val_acc:.4f}  '
          f'{time.time()-t0:.1f}s')

    # Early-stopping logic
    if val_acc > best_acc:
        best_acc = val_acc
        wait = 0
        torch.save(model.state_dict(), ckpt_stage1)
        print('  ↳ New best checkpoint saved.')
    else:
        wait += 1
        if wait >= PATIENCE:
            print('  ↳ Early stop Stage 1.')
            break

print(f'Stage 1 complete | best val_acc={best_acc:.4f}')

# ---------- 5. Stage 2 — freeze backbone, fine-tune head on tail classes ----------
print('\n========= Stage 2 (tail-class fine-tune) =========')

model2 = timm.create_model(model_name, pretrained=False,
                           num_classes=num_classes).to(device)
model2.load_state_dict(torch.load(ckpt_stage1, map_location=device))

# Freeze every parameter except the classifier head
for n, p in model2.named_parameters():
    p.requires_grad = ('head' in n)

criterion2 = nn.CrossEntropyLoss()
optimizer2 = optim.AdamW(filter(lambda p: p.requires_grad, model2.parameters()),
                         lr=1e-5, weight_decay=0.0)   # smaller LR
scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=5, gamma=0.1)

best_tail_acc, wait = 0.0, 0
ckpt_stage2 = 'vit_longtail_tailft.pth'

for epoch in range(1, STAGE2_EPOCHS + 1):
    t0 = time.time()
    model2.train()
    for x, y in tail_loader:
        x, y = x.to(device), y.to(device)
        optimizer2.zero_grad()
        loss = criterion2(model2(x), y)
        loss.backward()
        optimizer2.step()
    scheduler2.step()

    # Validate on the full validation set
    model2.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model2(x).argmax(dim=1)
            total   += y.size(0)
            correct += (pred == y).sum().item()
    val_acc = correct / total
    print(f'[Tail {epoch:02d}/{STAGE2_EPOCHS}] '
          f'val_acc={val_acc:.4f}  {time.time()-t0:.1f}s')

    if val_acc > best_tail_acc:
        best_tail_acc = val_acc
        wait = 0
        torch.save(model2.state_dict(), ckpt_stage2)
        print('  ↳ New best tail-fine-tune model saved.')
    else:
        wait += 1
        if wait >= PATIENCE:
            print('  ↳ Early stop Stage 2.')
            break

print(f'Finished! Stage 2 best val_acc={best_tail_acc:.4f}')