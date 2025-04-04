# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 23:29:21 2025

@author: MIS
"""

import cv2
import os
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import torchvision.utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Get path for dataset, train, val and test
datasets_dir = Path("datasets")
train_dir = datasets_dir / "train"
val_dir = datasets_dir / "val"
test_dir = datasets_dir / "test"

# Contrast never exceeds 80 which may also be problematic
# As contrast is important for distinguishing features, low contrast
# makes it more difficult for the model to differentiate textures.
# Features may look too similar, reducing the model’s ability to distinguish different terrain types.
# If contrast in real-world test images is higher, the model might not generalize well.
# Adjusts pixel values so that the output image's histogram is more uniform
# Spreads out the most frequent intensity values across the available range
def enhance_contrast(img):
    # Convert PIL Image to numpy array if needed
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Check if the image is grayscale or color
    if len(img.shape) == 2:
        # Image is already grayscale
        result = cv2.equalizeHist(img)
    
    elif len(img.shape) == 3:
        # For color images, convert to Lab color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        
        # Configure CLAHE with clip limit of 2 and tile grid size of (8, 8)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        
        # Apply CLAHE to the L channel only
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        # Convert back to RGB
        result = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    
    # Convert back to PIL Image if input was PIL
    if isinstance(img, Image.Image):
        result = Image.fromarray(result)
        
    return result

# Recall lab 1 - Unsharp masking allows control over the amount and radius of sharpening
# within the same image via Gaussian filtering
# The Laplacian method emphasizes all edges equally (even weak edges)
# and is therefore best suited to restoring missing details.
# Unsharp masking on the other hand enhances existing edges but doesn’t create new ones.
def sharpen_image(img, amount=0.5):
    # Convert PIL Image to numpy array if needed
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Create blurred version
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    
    # Apply unsharp mask
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    
    # Convert back to PIL Image if input was PIL
    if isinstance(img, Image.Image):
        sharpened = Image.fromarray(sharpened)
        
    return sharpened

class RandomOcclusion(nn.Module):

    def __init__(self, size_range=(10, 50), count_range=(1, 5), fill_value=0):

        super().__init__()
        self.size_range = size_range
        self.count_range = count_range
        self.fill_value = fill_value
        
    def forward(self, img):

        # Convert to tensor if it's a PIL Image
        if isinstance(img, Image.Image):
            img_tensor = F.to_tensor(img)
        else:
            img_tensor = img.clone()
            
        # Get dimensions
        _, height, width = img_tensor.shape
        
        # Determine number of occlusions
        num_occlusions = random.randint(self.count_range[0], self.count_range[1])
        
        # Apply occlusions
        for _ in range(num_occlusions):
            # Determine occlusion size
            h_size = random.randint(self.size_range[0], self.size_range[1])
            w_size = random.randint(self.size_range[0], self.size_range[1])
            
            # Determine occlusion location
            y = random.randint(0, height - h_size)
            x = random.randint(0, width - w_size)
            
            # Apply occlusion
            img_tensor[:, y:y + h_size, x:x + w_size] = self.fill_value
            
        # Convert back to PIL if input was PIL
        if isinstance(img, Image.Image):
            return F.to_pil_image(img_tensor)
        
        return img_tensor

# Custom transform for applying contrast enhancement and sharpening  
class EnhancementTransform(nn.Module):
    def __init__(self, sharpen_amount=0.3, apply_contrast=True):
        super().__init__()
        self.sharpen_amount = sharpen_amount
        self.apply_contrast = apply_contrast
        
    def forward(self, img):
        # Convert to numpy for OpenCV operations
        np_img = np.array(img)
        
        # Apply sharpening
        np_img = sharpen_image(np_img, self.sharpen_amount)
        
        # Apply contrast enhancement if enabled
        if self.apply_contrast:
            np_img = enhance_contrast(np_img)
            
        # Convert back to PIL Image
        return Image.fromarray(np_img.astype(np.uint8))

# Dataset class for loading and augmenting images
class ImageDataset(Dataset):
    def __init__(self, images_dir, transform=None, sample_size=None, random_seed=42):
        self.transform = transform
        
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Get list of class directories
        self.class_dirs = [d for d in Path(images_dir).iterdir() if d.is_dir()]
        
        # Create class indices mapping
        self.class_to_idx = {class_dir.name: i for i, class_dir in enumerate(sorted(self.class_dirs))}
        
        # Load images and labels
        self.imgs = []
        self.labels = []
        
        # Process each class
        for class_dir in sorted(self.class_dirs):
            class_name = class_dir.name
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files for this class
            image_paths = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            # Sample if needed
            if sample_size is not None and len(image_paths) > sample_size:
                image_paths = np.random.choice(image_paths, size=sample_size, replace=False)
            
            print(f"Processing {len(image_paths)} images for class '{class_name}'")
            
            # Store paths and labels
            for img_path in image_paths:
                self.imgs.append(str(img_path))
                self.labels.append(class_idx)
                
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = self.labels[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
    def __len__(self):
        return len(self.imgs)
    
class AugmentationGenerator:
    # 224×224 is a standard size used by many pre-trained CNN models like VGG16, ResNet50, etc.
    # These models were trained on ImageNet using this image size, so it became a common default.
    # 224×224 provides a good balance between detail preservation and processing speed (compared to larger sizes like 256×256).
    
    # Batch size - the number of samples (images in this case) that will be processed together in a single forward/backward pass through the system.
    # Smaller batches provide more noisy gradients which can sometimes help escape local minima
    # Larger batches provide more stable gradient estimates but may converge to sharper minima
    # Batch size of 32 is often considered a good compromise
    def __init__(self, img_height=224, img_width=224, sharpen_amount=0.3, contrast_enhancement=True,
                 use_occlusion=True, occlusion_size=(10, 30), occlusion_count=(1, 2)):
        self.img_height = img_height
        
        # Create the transformation pipeline for training data with augmentation
        # Create list of transforms
        train_transforms = [
            transforms.Resize((img_height, img_width)),
            EnhancementTransform(sharpen_amount, contrast_enhancement),
            transforms.Pad(padding=224, padding_mode='reflect'),
            transforms.RandomRotation(20),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.2, 0.2),  # Horizontal and vertical shift
                # Too much zoom could crop out important details.
                # Too little zoom might not introduce enough variation.
                # Zooming out too much may introduce excessive background, reducing the focus on key objects.

                scale=(0.9, 1.1),      # Zoom range
                fill=127,  # Let the padding mode handle it
            ),
            transforms.CenterCrop((224, 224)),
            transforms.ColorJitter(
                # Normally setting adjustment factor to be between 0.7 and 1.3 helps
                # keep brightness changes within a realistic range without making images too dark (unusable)
                # or too bright (washed out).
                # 0.7 darkens the image (simulating shadowy conditions).
                # 1.3 brightens the image (simulating strong lighting).
                # Too dark  = loss of detail.
                # Too bright = overexposure, washed-out colors.
                brightness=(0.7, 1.3),  # Brightness range
                contrast=0.1,           # Contrast variation
                saturation=0.1,         # Saturation variation
                hue=0.1                 # Hue variation
            ),
            # As the dataset is comprised of aerial views which include symmetric objects 
            # both horizontal and vertical flips are used
            # these flips can improve model robustness
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            
        ]
        
        # Add occlusion if enabled
        if use_occlusion:
            train_transforms.append(
                RandomOcclusion(size_range=occlusion_size, count_range=occlusion_count)
            )
            
        # Add final conversion and normalization
        train_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        # Create the transformation pipeline
        self.train_transform = transforms.Compose(train_transforms)
        
        # Basic transformation for validation/test without augmentations
        self.val_transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            EnhancementTransform(sharpen_amount, contrast_enhancement),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Create datasets for training, validation and optionally testing
    def create_datasets(self, train_dir, val_dir, test_dir=None, sample_size=None):

        train_dataset = ImageDataset(train_dir, transform=self.train_transform, sample_size=sample_size)
        val_dataset = ImageDataset(val_dir, transform=self.val_transform, sample_size=sample_size)
        
        if test_dir is not None:
            test_dataset = ImageDataset(test_dir, transform=self.val_transform)
            return train_dataset, val_dataset, test_dataset
        
        return train_dataset, val_dataset
    
    # Create data loaders for training, validation and optionally testing
    def create_dataloaders(self, train_dir, val_dir, test_dir=None, batch_size=32, sample_size=None):
        if test_dir is not None:
            train_dataset, val_dataset, test_dataset = self.create_datasets(
                train_dir, val_dir, test_dir, sample_size
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            return train_loader, val_loader, test_loader
        else:
            train_dataset, val_dataset = self.create_datasets(train_dir, val_dir, sample_size=sample_size)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            return train_loader, val_loader
        
def visualize_augmentations(image_path, n_augmentations=5, use_occlusion=True):

    # Load image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    # Apply enhancements first
    np_img = np.array(img)
    enhanced_img = enhance_contrast(sharpen_image(np_img))
    enhanced_pil = Image.fromarray(enhanced_img.astype(np.uint8))
    
    # Create list of transforms
    augment_transforms = [
        transforms.Resize((224, 224)),
        EnhancementTransform(0.3, True),
        transforms.Pad(padding=224, padding_mode='reflect'),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.9, 1.1), fill=127),
        transforms.CenterCrop((224, 224)),
        transforms.ColorJitter(brightness=(0.7, 1.3), contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
    
    # Add occlusion if enabled
    if use_occlusion:
        augment_transforms.append(
            RandomOcclusion(size_range=(10, 30), count_range=(1, 2))
        )
    
    # Create augmentation transform
    augment_transform = transforms.Compose(augment_transforms)
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(1, n_augmentations + 2, 1)
    plt.imshow(np_img)
    plt.title('Original')
    plt.axis('off')
    
    # Plot enhanced image
    plt.subplot(1, n_augmentations + 2, 2)
    plt.imshow(enhanced_img)
    plt.title('Enhanced')
    plt.axis('off')
    
    # Generate and plot augmented images
    for i in range(n_augmentations):
        # Apply augmentations
        augmented = augment_transform(enhanced_pil)
        
        # Convert to numpy for display
        augmented_np = np.array(augmented)
        
        # Display
        plt.subplot(1, n_augmentations + 2, i + 3)
        plt.imshow(augmented_np)
        plt.title(f'Augmented {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    print(augment_transforms)


if __name__ == "__main__":
    # This code runs when the script is executed directly
    # Example: Generate augmentations for a single image
    
    # Load an example image
    img_path = "datasets/train/Airport/121.jpg"
    if os.path.exists(img_path):
        # Visualize augmentations
        visualize_augmentations(img_path, n_augmentations=5, use_occlusion=True)


    else:
        print(f"Example image not found at {img_path}")