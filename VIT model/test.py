import os
import torch
import timm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
NUM_CLASSES  = 15
MODEL_NAME   = 'vit_base_patch16_224'
WEIGHTS_PATH = 'best_model.pth'                # .pth from your training
TEST_FOLDER  = r'testset/path'                 # raw string (Windows) or "testset/path"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True      # speed-up for fixed input size

# ------------------------------------------------------------------
# PRE-PROCESSING (must match training)
# ------------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ------------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------------
def load_model():
    model = timm.create_model(MODEL_NAME, pretrained=False,
                              num_classes=NUM_CLASSES)
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()
    return model

# ------------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader):
    y_true, y_pred = [], []
    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(imgs)
        preds  = logits.argmax(dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)
    return accuracy, y_true, y_pred

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == '__main__':
    # 1. dataset / loader
    test_ds = datasets.ImageFolder(TEST_FOLDER, transform=preprocess)
    test_loader = DataLoader(test_ds, batch_size=32,
                             shuffle=False, num_workers=4)

    # 2. model
    model = load_model()

    # 3. evaluate
    acc, y_true, y_pred = evaluate(model, test_loader)
    print(f'Test accuracy: {acc:.2%}\n')

    # 4. per-class report
    class_names = test_ds.classes          # e.g. ['airport','bridge',...]
    print(classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    ))
