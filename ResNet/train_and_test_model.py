import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models.resnet
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, mode='min'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return

        if self.mode == 'min':
            improvement = self.best_score - current_score
            if improvement > self.delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == 'max':
            improvement = current_score - self.best_score
            if improvement > self.delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

def trainModel(model_ver,train_loader,val_loader,datatype):
    train_losses = []
    val_losses = []
    num_epochs = 100
    torch.manual_seed(42)
    num_classes = 15

    if model_ver==18:
        model = models.resnet18(pretrained=True)
    elif model_ver==50:
        model = models.resnet50(pretrained=True)
    elif model_ver==101:
        model = models.resnet101(pretrained=True)

    early_stopper = EarlyStopping(patience=10, delta=0.01, mode='min')
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_classes = 15
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    
    trained = False
    bestweight_path = 'bestweight_resnet' + str(model_ver) +'_' + str(datatype) + '.pth'
    if os.path.exists(bestweight_path):
        bestweight = torch.load(bestweight_path)
        model.load_state_dict(bestweight)
        trained = True
    else:
        print("error: can't find bestweight!")
    
    if trained==False:
        for epoch in range(num_epochs):
            print(f"📚 Epoch {epoch}/{num_epochs}")
            model.train()
            train_loss, correct, total = 0.0, 0, 0
        
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
        
                outputs = model(images)
                loss = criterion(outputs, labels)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                train_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
            epoch_loss = train_loss / total
            epoch_acc = correct / total
            train_losses.append(train_loss)
            print(f"✅ Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            best_val_loss = 1.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
        
                    val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
        
            val_epoch_loss = val_loss / val_total
            val_epoch_acc = val_correct / val_total
            val_losses.append(val_loss)
            print(f"🧪 Val Loss: {val_epoch_loss:.4f} | Acc: {val_epoch_acc:.4f}")
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                torch.save(model.state_dict(), bestweight_path)

            early_stopper(val_loss)
            if early_stopper.early_stop:
                print(f"⛔ Early stopping at epoch {epoch+1}")
                break
    
        epochs = list(range(1, len(train_losses)+1))
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')
        plt.show()

def testModel(model_ver,test_loader,datatype):
    if model_ver==18:
        model = models.resnet18(pretrained=True)
    elif model_ver==50:
        model = models.resnet50(pretrained=True)
    elif model_ver==101:
        model = models.resnet101(pretrained=True)

    num_classes = 15
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    bestweight_path = 'bestweight_resnet' + str(model_ver) +'_' + str(datatype) + '.pth'
    if os.path.exists(bestweight_path):
        bestweight = torch.load(bestweight_path)
        model.load_state_dict(bestweight)
    else:
        print("error: can't find bestweight!")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
      for images, labels in test_loader:
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          _, preds = torch.max(outputs, 1)
    
          all_preds.extend(preds.cpu().numpy())
          all_labels.extend(labels.cpu().numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"✅ ResNet-{model_ver}, Accuracy: {accuracy * 100:.2f}%")
    
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
    
    print("🔍 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

def attention_map(model_ver,test_loader,datatype):
    if model_ver == 18:
        model = models.resnet18(pretrained=True)
    elif model_ver == 50:
        model = models.resnet50(pretrained=True)
    elif model_ver == 101:
        model = models.resnet101(pretrained=True)

    num_classes = 15
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    bestweight_path = f'bestweight_resnet{model_ver}_{datatype}.pth'
    if os.path.exists(bestweight_path):
        bestweight = torch.load(bestweight_path, map_location=device)
        model.load_state_dict(bestweight)
    else:
        print("error: can't find bestweight!")
        return

    model.eval()

    image, label = next(iter(test_loader))
    image = image[0].unsqueeze(0).to(device)
    label = label[0].unsqueeze(0).to(device)

    feature_map = []

    def hook_fn(module, input, output):
        feature_map.append(output)

    hook = model.layer4.register_forward_hook(hook_fn)

    output = model(image)
    _, preds = torch.max(output, 1)

    feature_map = feature_map[0].squeeze(0).cpu().detach().numpy()  # shape: [C, H, W]
    fc_weights = model.fc.weight[preds.item()].cpu().detach().numpy()  # shape: [C]

    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # shape: [H, W]
    for i, w in enumerate(fc_weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / cam.max()

    img = image[0].cpu().detach().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    plt.imshow(cam, alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.show()

    hook.remove()
    return cam