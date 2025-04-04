import cv2
import random
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    # Check if the image is grayscale or color
    if len(img.shape) == 2:
        # Image is already grayscale
        return cv2.equalizeHist(img)
    
    elif len(img.shape) == 3:
        # For color images, convert to Lab color space first
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        
        # Configure CLAHE
        # A clip-limit of 2 to 3 is normally considered a good place to start
        # tildgridsize of (8, 8) is also considered a good place to start
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
        
        # Apply CLAHE to the L channel only
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
        
        return enhanced

# Recall lab 1 - Unsharp masking allows control over the amount and radius of sharpening
# within the same image via Gaussian filtering
# The Laplacian method emphasizes all edges equally (even weak edges)
# and is therefore best suited to restoring missing details.
# Unsharp masking on the other hand enhances existing edges but doesn’t create new ones.
def sharpen_image(img, amount=0.5):
    # Create blurred version
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    
    # Apply unsharp mask
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)

    return sharpened

class AugmentationGenerator:
    # 224×224 is a standard size used by many pre-trained CNN models like VGG16, ResNet50, etc.
    # These models were trained on ImageNet using this image size, so it became a common default.
    # 224×224 provides a good balance between detail preservation and processing speed (compared to larger sizes like 256×256).
    
    # Batch size - the number of samples (images in this case) that will be processed together in a single forward/backward pass through the system.
    # Smaller batches provide more noisy gradients which can sometimes help escape local minima
    # Larger batches provide more stable gradient estimates but may converge to sharper minima
    # Batch size of 32 is often considered a good compromise
    def __init__(self, img_height=224, img_width=224, sharpen_amount=0.3, contrast_enhancement=True):
        self.img_height = img_height
        self.img_width = img_width
        
        # Create augmentation generator for training data
        self.augmentation_gen = ImageDataGenerator(
            rotation_range=20,  # Rotate up to 20 degrees
            width_shift_range=0.2,  # Horizontal shift
            height_shift_range=0.2,  # Vertical shift
            # Normally setting adjustment factor to be between 0.7 and 1.3 helps
            # keep brightness changes within a realistic range without making images too dark (unusable)
            # or too bright (washed out).
            # 0.7 darkens the image (simulating shadowy conditions).
            # 1.3 brightens the image (simulating strong lighting).
            # Too dark  = loss of detail.
            # Too bright = overexposure, washed-out colors.
            brightness_range=(0.7, 1.3),
            # Too much zoom could crop out important details.
            # Too little zoom might not introduce enough variation.
            # Zooming out too much may introduce excessive background, reducing the focus on key objects.
            zoom_range=[0.9, 1.1],
            # As the dataset is comprised of aerial views which include symmetric objects 
            # both horizontal and vertical flips are used
            # these flips can improve model robustness
            horizontal_flip=True,
            vertical_flip=True,  # Useful for aerial imagery
            fill_mode='reflect',
            channel_shift_range=0.1,         # subtle color variation
            shear_range=0.0                  # No shear (to avoid distortion)
        )
    
    def load_and_augment_images(self, images_dir, n_augmentations=3, sample_size=None, random_seed=42):

        np.random.seed(random_seed)
        
        images = []
        labels = []
        class_indices = {}
        
        # Get list of class directories
        class_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
        
        # Create class indices mapping
        for i, class_dir in enumerate(sorted(class_dirs)):
            class_indices[class_dir.name] = i
        
        # Process each class
        for class_dir in sorted(class_dirs):
            class_name = class_dir.name
            class_idx = class_indices[class_name]
            
            # Get all image files for this class
            image_paths = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            # Sample if needed
            if sample_size is not None and len(image_paths) > sample_size:
                image_paths = np.random.choice(image_paths, size=sample_size, replace=False)
            
            print(f"Processing {len(image_paths)} images for class '{class_name}'")
            
            # Load and process each image
            for img_path in image_paths:
                # Read and resize image
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_height, self.img_width))
                
                # Adjust image
                img = enhance_contrast(sharpen_image(img))
                
                # Add image
                images.append(img)
                labels.append(class_idx)
                
                # Generate augmentations
                if n_augmentations > 0:
                    img_array = np.expand_dims(img, axis=0)
                    aug_gen = self.augmentation_gen.flow(img_array, batch_size=1)
                    
                    for _ in range(n_augmentations):
                        aug_img = np.clip(next(aug_gen)[0], 0, 255).astype(np.uint8)
                        images.append(aug_img)
                        labels.append(class_idx)
        
        return images, labels, class_indices
    
def visualize_augmentations(image, n_augmentations=5):

    # Create augmentation generator
    generator = AugmentationGenerator()
    
    # Reshape image for the generator
    img_array = np.expand_dims(image, 0)
    
    # Create iterator
    aug_iter = generator.augmentation_gen.flow(img_array, batch_size=1)

    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(1, n_augmentations + 1, 1)
    plt.imshow(image)
    plt.title('Adjusted Original')
    plt.axis('off')

    # Generate and plot augmented images
    for i in range(n_augmentations):
        augmented = next(aug_iter)[0].astype(np.uint8)        
        plt.subplot(1, n_augmentations + 1, i + 2)
        plt.imshow(augmented)
        plt.title(f'Augmented {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # This code runs when the script is executed directly
    # Example: Generate augmentations for a single image
    
    # Load an example image
    img_path = "datasets/train/Airport/121.jpg"
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        # plot unadjusted original
        plt.imshow(img)
        plt.title('Original')
        plt.axis('off')

        # Adjust image
        img = enhance_contrast(sharpen_image(img))
        
        # Show enhancement examples
        plt.figure(figsize=(15, 5))
        
        # Visualize augmentations
        visualize_augmentations(img, n_augmentations=5)
        plt.tight_layout()
        plt.show()
        
    else:
        print(f"Example image not found at {img_path}")