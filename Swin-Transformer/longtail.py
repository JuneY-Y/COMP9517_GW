import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)

# Parameters
original_dir = Path("Aerial_Landscapes")
output_dir = Path("datasets_longtail")
train_ratio, val_ratio = 0.7, 0.15
imbalance_factor = 10  # The long-tail imbalance factor; larger means fewer samples in tail classes

# Step 1: Create the directory structure for train / val / test splits
for split in ['train', 'val', 'test']:
    (output_dir / split).mkdir(parents=True, exist_ok=True)

# Step 2: Perform an initial split into train / val / test
image_records = defaultdict(dict)

for class_dir in original_dir.iterdir():
    if class_dir.is_dir():
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(images)

        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        image_records[class_dir.name]['train'] = images[:n_train]
        image_records[class_dir.name]['val'] = images[n_train:n_train + n_val]
        image_records[class_dir.name]['test'] = images[n_train + n_val:]

        # Copy val and test images directly; keep train for sampling
        for split in ['val', 'test']:
            target_dir = output_dir / split / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for img_path in image_records[class_dir.name][split]:
                shutil.copy(img_path, target_dir / img_path.name)

# Step 3: Apply long-tail sampling to train set
class_names = sorted(image_records.keys())
class_sizes = {cls: len(image_records[cls]['train']) for cls in class_names}
max_count = max(class_sizes.values())
num_classes = len(class_names)

# Determine the desired sample count for each class using exponential decay
desired_train_counts = {
    cls: max(int(max_count * (1 / imbalance_factor) ** (rank / (num_classes - 1))), 1)
    for rank, cls in enumerate(sorted(class_names, key=lambda x: class_sizes[x], reverse=True))
}

# Sample and copy images for train set based on long-tail distribution
for cls in class_names:
    images = image_records[cls]['train']
    target_count = min(len(images), desired_train_counts[cls])
    sampled_images = random.sample(images, target_count)

    target_dir = output_dir / 'train' / cls
    target_dir.mkdir(parents=True, exist_ok=True)
    for img_path in sampled_images:
        shutil.copy(img_path, target_dir / img_path.name)

    print(f"[{cls}] original: {len(images)}, sampled: {len(sampled_images)}, val: {len(image_records[cls]['val'])}, test: {len(image_records[cls]['test'])}")

print("Dataset split with long-tail train set complete.")
