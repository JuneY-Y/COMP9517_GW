import os
import time
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


LEVEL_CONFIG = {
    1: {  
        "geo_prob": 0.3,
        "angle_range": 15,
        "color_prob": 0.3,
        "bright_range": (0.9, 1.1),
        "contrast_range": (0.9, 1.1),
        "saturation_range": (0.9, 1.1),
        "hue": 0.05,
        "noise_prob": 0.3,
        "noise_std": 5
    },
    2: { 
        "geo_prob": 0.5,
        "angle_range": 30,
        "color_prob": 0.5,
        "bright_range": (0.8, 1.2),
        "contrast_range": (0.8, 1.2),
        "saturation_range": (0.8, 1.2),
        "hue": 0.1,
        "noise_prob": 0.5,
        "noise_std": 15
    },
    3: {  
        "geo_prob": 0.8,
        "angle_range": 45,
        "color_prob": 0.8,
        "bright_range": (0.7, 1.3),
        "contrast_range": (0.7, 1.3),
        "saturation_range": (0.7, 1.3),
        "hue": 0.2,
        "noise_prob": 0.8,
        "noise_std": 25
    }
}

def add_noise(img: Image.Image, std: float) -> Image.Image:

    np_img = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std, np_img.shape)
    np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)

def augment_image(img: Image.Image, level: int = 2) -> Image.Image:

    cfg = LEVEL_CONFIG[level]
    aug = img.copy()

    if random.random() < cfg["geo_prob"]:
        angle = random.uniform(-cfg["angle_range"], cfg["angle_range"])
        aug = aug.rotate(angle, expand=True)
    if random.random() < cfg["geo_prob"]:
        aug = aug.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < cfg["geo_prob"]:
        aug = aug.transpose(Image.FLIP_TOP_BOTTOM)

  
    if random.random() < cfg["color_prob"]:
        jitter = transforms.ColorJitter(
            brightness=cfg["bright_range"],
            contrast=cfg["contrast_range"],
            saturation=cfg["saturation_range"],
            hue=cfg["hue"]
        )
        aug = jitter(aug)


    if random.random() < cfg["noise_prob"]:
        aug = add_noise(aug, std=cfg["noise_std"])

    return aug

base_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='datasets/train', transform=base_transform)
val_dataset   = datasets.ImageFolder(root='datasets/test',  transform=base_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4)


num_classes = len(train_dataset.classes)
model = timm.create_model(
    'vit_base_patch16_224',
    pretrained=False,
    checkpoint_path='B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
    num_classes=num_classes
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs       = 50
best_val_acc    = 0.0
patience        = 5
patience_counter = 0
save_path       = 'aug_model.pth'

λ_cons = 0.5 


start_time = time.time()
for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)


        batch_pil = [ transforms.ToPILImage()(x.cpu()) for x in inputs ]
        levels    = [ random.choice([1,2,3]) for _ in batch_pil ]
        batch_aug = [ augment_image(p, level=l) for p, l in zip(batch_pil, levels) ]
        inputs_aug = torch.stack([ base_transform(p) for p in batch_aug ]).to(device)


        logits_orig = model(inputs)
        logits_aug  = model(inputs_aug)

        loss_ce   = criterion(logits_aug, labels)
        p_orig    = F.softmax(logits_orig, dim=1)
        p_aug     = F.softmax(logits_aug,  dim=1)
        loss_cons = F.kl_div(p_aug.log(), p_orig, reduction='batchmean')

        loss = loss_ce + λ_cons * loss_cons

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    scheduler.step()


    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total  += labels.size(0)
            correct += (preds == labels).sum().item()
    val_acc = correct / total

    epoch_time = time.time() - epoch_start
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"- Loss: {epoch_loss:.4f} "
          f"- Val Acc: {val_acc:.4f} "
          f"- Time: {epoch_time:.1f}s")


    if val_acc > best_val_acc:
        best_val_acc    = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), save_path)
        print(f"  --> New best model (Acc: {val_acc:.4f}), saved to {save_path}")
    else:
        patience_counter += 1
        print(f"  --> No improvement for {patience_counter} epoch(s)")

    if patience_counter >= patience:
        print(f"Early stopping after {epoch+1} epochs.")
        break

total_time = time.time() - start_time
print(f"Total training time: {total_time:.1f}s")