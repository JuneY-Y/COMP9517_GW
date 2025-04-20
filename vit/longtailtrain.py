
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import timm
import wandb

def main():
    
    wandb.init(project="ViT-aerial-classification", config={
        "epochs": 50,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "model": "vit_base_patch16_224",
        "dataset": "your-dataset"
    })
    config = wandb.config


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),        
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root='9517longtail/train', transform=transform)
    val_dataset   = datasets.ImageFolder(root='9517longtail/val', transform=transform)
    test_dataset  = datasets.ImageFolder(root='9517longtail/test', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)


    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    state_dict = torch.load('pytorch_model.bin', map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    num_features = model.head.in_features
    model.head = nn.Linear(num_features, 15)
    model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    num_epochs = config.epochs

    for epoch in range(num_epochs):
        epoch_start = time.time()  
        model.train()
        running_loss = 0.0


        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        epoch_duration = time.time() - epoch_start


        remaining_epochs = num_epochs - (epoch + 1)
        total_remaining_time = remaining_epochs * epoch_duration

        rem_minutes = int(total_remaining_time // 60)
        rem_seconds = int(total_remaining_time % 60)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Time: {epoch_duration:.2f} sec')
        print(f'Estimated remaining time: {rem_minutes} min {rem_seconds} sec')

   
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
        val_accuracy = val_correct / val_total
        print(f'Epoch {epoch+1}/{num_epochs}, Val Accuracy: {val_accuracy:.4f}')

 

        wandb.log({
            "epoch": epoch+1,
            "training_loss": epoch_loss,
            "val_accuracy": val_accuracy,

            "epoch_duration_sec": epoch_duration,
            "total_remaining_time_sec": total_remaining_time
        })
        print()  
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (preds == labels).sum().item()
    test_accuracy = test_correct / test_total
    print(f'Epoch {epoch+1}/{num_epochs}, Test Accuracy: {test_accuracy:.4f}')


    torch.save(model.state_dict(), "vit_model.pth")
    wandb.save("vit_model.pth")

if __name__ == '__main__':
    main()
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from torchvision.datasets import ImageFolder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('vit_base_patch16_224', pretrained=False)
num_features = model.head.in_features
model.head = nn.Linear(num_features, 15)  
model = model.to(device)


state_dict = torch.load("vit_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),           
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


img_path = "datasets"  
image = Image.open(img_path).convert("RGB")
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  

with torch.no_grad():
    outputs = model(input_batch)
    probabilities = torch.softmax(outputs, dim=1)
    pred_class = torch.argmax(probabilities, dim=1)

test_dataset = ImageFolder(root='datasets/test', transform=preprocess)
class_names = test_dataset.classes
pred_name = class_names[pred_class.item()]
print("预测类别名称：", pred_name)

