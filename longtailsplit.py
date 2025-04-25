"""
create_longtail_split.py
------------------------
Split a balanced Aerial Landscape dataset into train/val/test folders, then
apply a power-law (long-tailed) sampling to the *training* split.

• 70 % train / 15 % val / 15 % test by default
• `imbalance_factor` controls how severe the long tail is
• Reproducible with a fixed random seed (42)

Folder layout after running:
datasets_longtail/
    ├── train/   # long-tailed distribution
    ├── val/     # balanced
    └── test/    # balanced
"""

from pathlib import Path
from collections import defaultdict
import random, shutil, os

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
random.seed(42)

SRC_DIR   = Path("Aerial_Landscapes")   # original balanced dataset
OUT_DIR   = Path("datasets_longtail")   # target directory
TRAIN_R, VAL_R = 0.70, 0.15             # split ratios
IMBALANCE_FACTOR = 10                   # the larger, the rarer the tail classes

# ------------------------------------------------------------------
# 1. Create base folder tree
# ------------------------------------------------------------------
for split in ["train", "val", "test"]:
    (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 2. First do a *balanced* train/val/test split
# ------------------------------------------------------------------
records = defaultdict(dict)   # {class: {'train': [...], 'val': [...], 'test': [...]}}

for class_dir in SRC_DIR.iterdir():
    if class_dir.is_dir():
        imgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(imgs)

        n_total = len(imgs)
        n_train = int(TRAIN_R * n_total)
        n_val   = int(VAL_R   * n_total)

        records[class_dir.name]['train'] = imgs[:n_train]
        records[class_dir.name]['val']   = imgs[n_train:n_train + n_val]
        records[class_dir.name]['test']  = imgs[n_train + n_val:]

        # Copy val & test now (train will be down-sampled later)
        for split in ['val', 'test']:
            target = OUT_DIR / split / class_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for img in records[class_dir.name][split]:
                shutil.copy(img, target / img.name)

# ------------------------------------------------------------------
# 3. Long-tail sampling on the train split
# ------------------------------------------------------------------
class_names  = sorted(records.keys())
class_sizes  = {c: len(records[c]['train']) for c in class_names}
max_count    = max(class_sizes.values())
num_classes  = len(class_names)

# Power-law allocation: head ≈ max_count, tail ≈ max_count / IMBALANCE_FACTOR
desired_train_counts = {
    cls: max(int(max_count * (1 / IMBALANCE_FACTOR) ** (rank / (num_classes - 1))), 1)
    for rank, cls in enumerate(
        sorted(class_names, key=lambda x: class_sizes[x], reverse=True)
    )
}

for cls in class_names:
    images        = records[cls]['train']
    target_n      = min(len(images), desired_train_counts[cls])
    sampled_imgs  = random.sample(images, target_n)

    target_dir = OUT_DIR / 'train' / cls
    target_dir.mkdir(parents=True, exist_ok=True)
    for img in sampled_imgs:
        shutil.copy(img, target_dir / img.name)

    print(f"[{cls}] original: {len(images)}, sampled: {len(sampled_imgs)}, "
          f"val: {len(records[cls]['val'])}, test: {len(records[cls]['test'])}")

print("✅ Finished: long-tailed training set created.")