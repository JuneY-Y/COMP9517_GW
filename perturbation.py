import os, random, numpy as np
from PIL import Image
import torchvision.transforms as transforms

# -------------------------------------------------------------------
# 1. Three-level augmentation presets
# -------------------------------------------------------------------
LEVEL_CONFIG = {
    1: {  # light augmentation
        "geo_prob": 0.30,
        "angle_range": 15,
        "color_prob": 0.30,
        "bright_range": (0.9, 1.1),
        "contrast_range": (0.9, 1.1),
        "saturation_range": (0.9, 1.1),
        "hue": 0.05,
        "noise_prob": 0.30,
        "noise_std": 5,
    },
    2: {  # medium augmentation  (matches the training script default)
        "geo_prob": 0.50,
        "angle_range": 30,
        "color_prob": 0.50,
        "bright_range": (0.8, 1.2),
        "contrast_range": (0.8, 1.2),
        "saturation_range": (0.8, 1.2),
        "hue": 0.10,
        "noise_prob": 0.50,
        "noise_std": 15,
    },
    3: {  # heavy augmentation
        "geo_prob": 0.80,
        "angle_range": 45,
        "color_prob": 0.80,
        "bright_range": (0.7, 1.3),
        "contrast_range": (0.7, 1.3),
        "saturation_range": (0.7, 1.3),
        "hue": 0.20,
        "noise_prob": 0.80,
        "noise_std": 25,
    },
}

# -------------------------------------------------------------------
# 2.  Helper: add Gaussian noise
# -------------------------------------------------------------------
def add_noise(img: Image.Image, std: float) -> Image.Image:
    """Add Gaussian noise with standard deviation `std` to an RGB image."""
    arr   = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std, arr.shape)
    arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

# -------------------------------------------------------------------
# 3.  Apply one random augmentation level to a single PIL image
# -------------------------------------------------------------------
def augment_image(img: Image.Image, level: int = 2) -> Image.Image:
    """
    Parameters
    ----------
    img   : PIL.Image
    level : int   {1|2|3}. 1 = weak, 2 = medium, 3 = strong.
    """
    cfg = LEVEL_CONFIG[level]
    out = img.copy()

    # -- geometric transforms ----------------------------------------
    if random.random() < cfg["geo_prob"]:
        angle = random.uniform(-cfg["angle_range"], cfg["angle_range"])
        out   = out.rotate(angle, expand=True)
    if random.random() < cfg["geo_prob"]:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < cfg["geo_prob"]:
        out = out.transpose(Image.FLIP_TOP_BOTTOM)

    # -- colour jitter ------------------------------------------------
    if random.random() < cfg["color_prob"]:
        jitter = transforms.ColorJitter(
            brightness = cfg["bright_range"],
            contrast   = cfg["contrast_range"],
            saturation = cfg["saturation_range"],
            hue        = cfg["hue"],
        )
        out = jitter(out)

    # -- Gaussian noise ----------------------------------------------
    if random.random() < cfg["noise_prob"]:
        out = add_noise(out, std=cfg["noise_std"])

    return out

# -------------------------------------------------------------------
# 4.  Loop over an ImageFolder-style directory and save augmented JPGs
# -------------------------------------------------------------------
def process_dataset_folder(input_root: str, output_root: str, level: int = 2) -> None:
    """
    Recursively iterate through `input_root`, apply augment_image(), and
    write new files to a mirrored folder tree rooted at `output_root`.

    Output filename format: <original_name>_L<level>.jpg
    """
    for dir_path, _, files in os.walk(input_root):
        rel_subdir = os.path.relpath(dir_path, input_root)
        target_dir = os.path.join(output_root, rel_subdir)
        os.makedirs(target_dir, exist_ok=True)

        for fname in files:
            if not fname.lower().endswith(".jpg"):
                continue
            src = os.path.join(dir_path, fname)
            try:
                img = Image.open(src).convert("RGB")
            except Exception:
                continue

            aug  = augment_image(img, level)
            stem, ext = os.path.splitext(fname)
            dst  = os.path.join(target_dir, f"{stem}_L{level}{ext}")
            aug.save(dst, "JPEG")
            print(f"Saved {dst}")

# -------------------------------------------------------------------
# 5.  Example CLI usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    input_dataset  = "/path/to/train"
    output_dataset = "/path/to/train_aug"

    # Choose one or more levels to generate:
    process_dataset_folder(input_dataset, output_dataset, level=1)  # light
    process_dataset_folder(input_dataset, output_dataset, level=2)  # medium
    process_dataset_folder(input_dataset, output_dataset, level=3)  # heavy