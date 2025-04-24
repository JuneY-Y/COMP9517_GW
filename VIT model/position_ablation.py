import torch
import timm
from torchvision import transforms, datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader

num_classes = 15                    
model_name = 'vit_base_patch16_224'    
weights_path = '/Users/yaogunzhishen/Desktop/未命名文件夹 7/best_vit_model.pth' 
test_folder = '/Users/yaogunzhishen/Desktop/未命名文件夹 7/datasets/test' 

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def load_model(with_pe=True):

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    if not with_pe and hasattr(model, 'pos_embed'):
        model.pos_embed.data.zero_()
    model.eval()
    return model


def evaluate_model(model, data_loader):

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy


if __name__ == '__main__':

    test_dataset = datasets.ImageFolder(root=test_folder, transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    

    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    

    model_with_pe = load_model(with_pe=True)
    model_without_pe = load_model(with_pe=False)

    acc_with_pe = evaluate_model(model_with_pe, test_loader)
    acc_without_pe = evaluate_model(model_without_pe, test_loader)
    
    print("use pos_embed: {:.2%}".format(acc_with_pe))
    print("no pos_embed: {:.2%}".format(acc_without_pe))