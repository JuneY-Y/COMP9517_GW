import time
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='datasets/train', transform=train_transforms)
val_dataset   = datasets.ImageFolder(root='datasets/test', transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


num_classes = len(train_dataset.classes)
model = timm.create_model('vit_base_patch16_224', pretrained=False, 
                          checkpoint_path='B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
                          num_classes=num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 50  


best_val_acc = 0.0      
patience = 5            
patience_counter = 0      
save_path = 'best_model.pth'  

start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()  

    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_time = time.time() - epoch_start_time

    scheduler.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    val_acc = correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Validation Acc: {val_acc:.4f}, Epoch Time: {epoch_time:.2f}s")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), save_path)
        print(f" --> New best validation accuracy! Model weights saved to {save_path}")
    else:
        patience_counter += 1
        print(f" --> No improvement for {patience_counter} epoch(s).")
    
    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

total_time = time.time() - start_time
print(f"Total Training Time: {total_time:.2f}s")