import torch
import timm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report


num_classes   = 15
model_name    = 'vit_base_patch16_224'
weights_path  = 'best_model.pth'
test_folder   = 'testset\path'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model():

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return acc, all_labels, all_preds

if __name__ == '__main__':
    test_dataset = datasets.ImageFolder(root=test_folder, transform=preprocess)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = load_model()
    acc, y_true, y_pred = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {acc:.2%}\n")