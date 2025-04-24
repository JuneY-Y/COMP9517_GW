# ==============================================================================
# name: split_dataset.py
# aim: Split aerial landscape classification dataset into train/val/test folders
# ==============================================================================

import os
import shutil
import random
from pathlib import Path

# split_dataset.py
random.seed(42)

original_dir = Path("Aerial_Landscapes")
output_dir = Path("datasets")
train_ratio, val_ratio = 0.7, 0.15

for split in ['train', 'val', 'test']:
    (output_dir / split).mkdir(parents=True, exist_ok=True)

for class_dir in original_dir.iterdir():
    if class_dir.is_dir():
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(images)

        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        split_dict = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        for split, imgs in split_dict.items():
            target_dir = output_dir / split / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for img_path in imgs:
                shutil.copy(img_path, target_dir / img_path.name)

print("Dataset split complete.")