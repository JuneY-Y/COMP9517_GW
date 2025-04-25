import torch
import timm
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F   # kept in case you extend the block later

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
NUM_CLASSES  = 15
MODEL_NAME   = 'vit_base_patch16_224'
WEIGHTS_PATH = '/Users/yaogunzhishen/Desktop/未命名文件夹 7/best_vit_model.pth'
TEST_FOLDER  = '/Users/yaogunzhishen/Desktop/未命名文件夹 7/datasets/test'

# Use Apple-Silicon GPU (MPS) if available, otherwise CPU
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('Using device:', DEVICE)

# ------------------------------------------------------------------
# Pre-processing (must match training)
# ------------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ------------------------------------------------------------------
# Load the full, un-ablated model
# ------------------------------------------------------------------
def load_model():
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()
    return model

# ------------------------------------------------------------------
# Block wrapper that **removes both residual skip connections**
# ------------------------------------------------------------------
class ResidualAblatedBlock(nn.Module):
    """
    The forward pass below drops the two residual skips inside a ViT block.
    Only the Attn and MLP branches remain. Accuracy will likely decrease.
    """
    def __init__(self, original_block):
        super().__init__()
        self.norm1 = original_block.norm1
        self.attn  = original_block.attn
        self.drop_path = original_block.drop_path    # Identity if rate == 0
        self.norm2 = original_block.norm2
        self.mlp   = original_block.mlp

    def forward(self, x):
        # --- first sub-layer (Attention) ---
        x1 = self.attn(self.norm1(x))
        x1 = self.drop_path(x1)          # no residual add

        # feed x1 to second sub-layer
        x2 = self.mlp(self.norm2(x1))
        x2 = self.drop_path(x2)          # no residual add

        return x2                        # final output (residual removed)

# ------------------------------------------------------------------
# Replace a range of blocks with ResidualAblatedBlock
# ------------------------------------------------------------------
def ablate_residual_range(model, start_idx, end_idx):
    """
    Replace blocks[start_idx : end_idx] (end exclusive) with residual-free
    versions. Works only if the model has a 'blocks' attribute.
    """
    if hasattr(model, 'blocks'):
        for i in range(start_idx, end_idx):
            print(f'Removing both residual skips in transformer block {i}')
            model.blocks[i] = ResidualAblatedBlock(model.blocks[i])
    else:
        print("Model has no attribute 'blocks'; cannot ablate residuals.")
    return model

# ------------------------------------------------------------------
# Simple accuracy evaluator
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(model, loader):
    total, correct = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).argmax(dim=1)
        total   += labels.size(0)
        correct += (preds == labels).sum().item()
    return correct / total

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == '__main__':
    # Dataset / dataloader
    test_ds = datasets.ImageFolder(TEST_FOLDER, transform=preprocess)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    # 1) Full model
    model_full = load_model()
    acc_full   = evaluate_model(model_full, test_loader)
    print('\n=== Full (baseline) model ===')
    print(f'Accuracy: {acc_full:.2%}')

    # Check how many transformer blocks exist
    num_blocks = len(model_full.blocks)
    print('\nTotal transformer blocks:', num_blocks)

    # 2) Residual-ablation experiments
    #    Three groups: blocks 0, 7, and 11 (half-open ranges)
    groups = [(0, 1), (7, 8), (11, 12)]

    for gid, (start, end) in enumerate(groups, 1):
        print(f'\n=== Ablation #{gid}: remove residuals in blocks {start}–{end-1} ===')
        model_ablate = load_model()                      # fresh copy
        model_ablate = ablate_residual_range(model_ablate, start, end)
        acc_ablate   = evaluate_model(model_ablate, test_loader)
        print(f'Accuracy after ablation {gid}: {acc_ablate:.2%}')