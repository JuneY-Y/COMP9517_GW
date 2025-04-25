import torch
import timm
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F  # (kept in case you add custom ops)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
num_classes  = 15
model_name   = 'vit_base_patch16_224'
weights_path = '/Users/yaogunzhishen/Desktop/未命名文件夹 7/best_vit_model.pth'   # update if needed
test_folder  = '/Users/yaogunzhishen/Desktop/未命名文件夹 7/datasets/test'

# Use Apple Silicon GPU if available, else CPU / CUDA
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
print("Using device:", device)

# --------------------------------------------------
# Pre-processing pipeline (same as training)
# --------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --------------------------------------------------
# Helper: load model & optionally ablate transformer blocks
# --------------------------------------------------
def load_model_transformer_ablation(disable_transformer: bool = False,
                                    disable_range: tuple | None = None):
    """
    Parameters
    ----------
    disable_transformer : bool
        If True, replace blocks[disable_range[0]:disable_range[1]] with nn.Identity().
    disable_range : (start, end)
        Half-open interval of block indices to disable (end is *exclusive*).
    """
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Ablate specific transformer blocks if requested
    if disable_transformer and hasattr(model, 'blocks'):
        print(f"Transformer ablation: disabling layers {disable_range[0]} "
              f"to {disable_range[1]-1} (inclusive)")
        for i in range(disable_range[0], disable_range[1]):
            model.blocks[i] = nn.Identity()

    model.eval()
    return model

# --------------------------------------------------
# Helper: evaluate accuracy on a DataLoader
# --------------------------------------------------
def evaluate_model(model, data_loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            total   += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total

# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == '__main__':
    # Load test dataset
    test_dataset = datasets.ImageFolder(root=test_folder, transform=preprocess)
    test_loader  = DataLoader(test_dataset, batch_size=32,
                              shuffle=False, num_workers=4)

    # -------- Full (non-ablated) model --------
    model_full = load_model_transformer_ablation(disable_transformer=False)
    acc_full   = evaluate_model(model_full, test_loader)
    print("\n=== Full model ===")
    print(f"Accuracy: {acc_full:.2%}")

    # How many transformer blocks does the backbone have?
    num_blocks = len(model_full.blocks)
    print("\nTotal transformer blocks:", num_blocks)

    # -------- Ablation experiments --------
    # Example: disable blocks [0], [7], [11]
    groups = [(0, 1), (7, 8), (11, 12)]

    for group_id, (start, end) in enumerate(groups, 1):
        print(f"\n=== Ablation run {group_id}: blocks {start}–{end-1} ===")
        model_ablate = load_model_transformer_ablation(
            disable_transformer=True, disable_range=(start, end)
        )
        acc_ablate = evaluate_model(model_ablate, test_loader)
        print(f"Accuracy after disabling group {group_id}: {acc_ablate:.2%}")