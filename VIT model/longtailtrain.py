# train_vit.py
import time, torch, timm, wandb
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------------------------------------------------
# 0.  Hyper-parameters (edit here or override from CLI / wandb)
# -------------------------------------------------------------
CFG = dict(
    project        = "ViT-aerial-classification",
    run_name       = "vit-base-longtail",
    epochs         = 50,
    lr             = 1e-4,
    batch_size     = 32,
    model_name     = "vit_base_patch16_224",
    num_classes    = 15,
    train_dir      = "9517longtail/train",
    val_dir        = "9517longtail/val",
    test_dir       = "9517longtail/test",
    init_ckpt      = "pytorch_model.bin",   # pretrained ViT weights
)

# -------------------------------------------------------------
# 1.  Init Weights & Biases
# -------------------------------------------------------------
wandb.init(project=CFG["project"], name=CFG["run_name"], config=CFG)
cfg = wandb.config

# -------------------------------------------------------------
# 2.  Device
# -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------------------
# 3.  Transforms & Dataloaders
# -------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(cfg.train_dir, transform)
val_ds   = datasets.ImageFolder(cfg.val_dir,   transform)
test_ds  = datasets.ImageFolder(cfg.test_dir,  transform)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                          shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                          shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size,
                          shuffle=False, num_workers=4)

# -------------------------------------------------------------
# 4.  Model: ViT-Base + new classifier head
# -------------------------------------------------------------
model = timm.create_model(cfg.model_name, pretrained=False)
# **FIX**: `torch.load(..., weights_only=True)` is only valid for `torch.load_state_dict_from_url`;
# just load and ignore missing keys
state_dict = torch.load(cfg.init_ckpt, map_location=device)
model.load_state_dict(state_dict, strict=False)     # ignore head mismatch
model.head = nn.Linear(model.head.in_features, cfg.num_classes)
model.to(device)

# -------------------------------------------------------------
# 5.  Loss / Optimiser
# -------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

# -------------------------------------------------------------
# 6.  Training loop
# -------------------------------------------------------------
best_val_acc = 0.0

for epoch in range(cfg.epochs):
    t0 = time.time()
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    train_loss = running_loss / len(train_ds)
    epoch_time = time.time() - t0

    # ---------- validation ----------
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
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_acc": val_acc,
        "epoch_time": epoch_time,
        "time_left_est": epoch_time * (cfg.epochs - epoch - 1)
    })

    print(f"[{epoch+1:02d}/{cfg.epochs}] "
          f"loss={train_loss:.4f}  val_acc={val_acc:.4f}  "
          f"time={epoch_time:.1f}s")

    # save best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "vit_best.pth")

# -------------------------------------------------------------
# 7.  Test set evaluation
# -------------------------------------------------------------
model.load_state_dict(torch.load("vit_best.pth", map_location=device))
model.eval()
correct = total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
test_acc = correct / total
print(f"Final test accuracy: {test_acc:.4f}")
wandb.log({"test_acc": test_acc})

# -------------------------------------------------------------
# 8.  Save final model artifact
# -------------------------------------------------------------
torch.save(model.state_dict(), "vit_final.pth")
wandb.save("vit_final.pth")
wandb.finish()