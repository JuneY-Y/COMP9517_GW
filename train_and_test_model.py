import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models.resnet
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, mode='min'):
        """
        参数说明：
        - patience: 容忍多少次没有提升后停止
        - delta: 最小提升量（小于这个不算提升）
        - mode: 'min' 表示监控 loss，'max' 表示监控 acc
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score):
        # 初次
        if self.best_score is None:
            self.best_score = current_score
            return

        # 判断是否是提升
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
    
    # 冻结所有层的参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 替换 fc 层
    num_classes = 15  # 设置你任务的类别数量
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 设置设备（使用 CUDA 如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    
    # 训练设置
    trained = False
    bestweight_path = 'bestweight_resnet' + str(model_ver) +'_' + str(datatype) + '.pth'
    if os.path.exists(bestweight_path):
        bestweight = torch.load(bestweight_path)
        model.load_state_dict(bestweight)
        trained = True
    else:
        print("error: can't find bestweight!")
    
    # 开始训练
    if trained==False:
        for epoch in range(num_epochs):
            print(f"📚 Epoch {epoch}/{num_epochs}")
            model.train()  # 设置为训练模式
            train_loss, correct, total = 0.0, 0, 0
        
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
        
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)
        
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                # 统计信息
                train_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
            epoch_loss = train_loss / total
            epoch_acc = correct / total
            train_losses.append(train_loss)
            print(f"✅ Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        
            # 评估模型
            model.eval()  # 设置为评估模式
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

    num_classes = 15  # 设置你任务的类别数量
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
    
    # 计算准确率
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"✅ ResNet-{model_ver}, Accuracy: {accuracy * 100:.2f}%")
    
    # 输出 classification_report
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
    
    # 输出混淆矩阵
    print("🔍 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
