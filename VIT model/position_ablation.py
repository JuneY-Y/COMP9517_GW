"""
vit_positional_embedding_ablation.py
------------------------------------
Compares ViT accuracy with and without positional-embedding information.
"""

import torch
import timm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------
NUM_CLASSES  = 15
MODEL_NAME   = "vit_base_patch16_224"
WEIGHTS_PATH = "/Users/yaogunzhishen/Desktop/Folder7/best_vit_model.pth"
TEST_FOLDER  = "/Users/yaogunzhishen/Desktop/Folder7/datasets/test"

# Prefer Apple-Silicon (MPS) if available; otherwise CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", DEVICE)

# --------------------------------------------------------------------
# DATA PRE-PROCESSING (must match training)
# --------------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# --------------------------------------------------------------------
# HELPER ─ load ViT and optionally zero-out positional embeddings
# --------------------------------------------------------------------
def load_model(with_pe: bool = True):
    """
    Parameters
    ----------
    with_pe : bool
        If False, the learned positional-embedding tensor `pos_embed`
        is replaced with zeros, effectively removing spatial information.
    """
    model = timm.create_model(MODEL_NAME, pretrained=False,
                              num_classes=NUM_CLASSES)
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state)

    if not with_pe and hasattr(model, "pos_embed"):
        model.pos_embed.data.zero_()      # erase positional information

    model = model.to(DEVICE).eval()
    return model

# --------------------------------------------------------------------
# HELPER ─ quick Top-1 accuracy
# --------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader):
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total

# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    test_ds = datasets.ImageFolder(TEST_FOLDER, transform=preprocess)
    test_loader = DataLoader(test_ds, batch_size=32,
                             shuffle=False, num_workers=4)

    # ▶ baseline (with positional embeddings)
    model_pe   = load_model(with_pe=True)
    acc_pe     = evaluate(model_pe, test_loader)

    # ▶ ablation (positional embeddings set to zero)
    model_nope = load_model(with_pe=False)
    acc_nope   = evaluate(model_nope, test_loader)

    print(f"With pos-embed : {acc_pe:.2%}")
    print(f"No  pos-embed : {acc_nope:.2%}")