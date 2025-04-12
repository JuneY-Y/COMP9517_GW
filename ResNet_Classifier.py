import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models.resnet
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def extract_features(model,loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            output = model(imgs)  # 输出的特征向量维度通常为 2048（ResNet50 默认）
            features.append(output.cpu())  # 留在 torch 格式
            labels.extend(lbls.numpy())    # 先放 numpy，再转成 tensor
    features = torch.cat(features, dim=0)
    labels = torch.tensor(labels)
    return features, labels
    #         features.append(output.cpu().numpy())
    #         labels.extend(lbls.numpy())
    # features = np.concatenate(features, axis=0)
    # labels = np.array(labels)
    # return features, labels

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, num_classes=15, dropout=0.5):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_MLP(train_features, train_labels, test_features, num_classes=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPClassifier(input_dim=train_features.shape[1], num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_train = torch.tensor(train_features, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_labels, dtype=torch.long).to(device)
    X_test = torch.tensor(test_features, dtype=torch.float32).to(device)
    
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1).cpu().numpy()
    return preds

class ProtoNetClassifier:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.prototypes = None  # 原型向量，每类一个

    def compute_prototypes(self, features, labels):
        """
        根据训练特征和标签计算每个类的原型（均值）
        """
        prototypes = []
        for c in range(self.n_classes):
            class_features = features[labels == c]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        self.prototypes = torch.stack(prototypes)

    def predict(self, features):
        """
        根据欧氏距离判断所属类别
        """
        dists = torch.cdist(features, self.prototypes)  # 计算距离 [N, C]
        preds = dists.argmin(dim=1)
        return preds

def classifierTrain(model_ver, train_loader, test_loader, classifier):
    if model_ver==18:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_ver==50:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif model_ver==101:
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        
    model.fc = nn.Identity()  # 用 Identity 替换全连接层，输出即为特征
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  # 关闭 Dropout、BatchNorm 的训练行为
    torch.manual_seed(42)
    
    train_features, train_labels = extract_features(model,train_loader)
    test_features, test_labels = extract_features(model,test_loader)

    if classifier=='SVM':
        svm_clf = SVC(kernel='rbf', C=9, gamma='scale')
        svm_clf.fit(train_features, train_labels)
        predictions = svm_clf.predict(test_features)
    elif classifier=='KNN':
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(train_features, train_labels)
        predictions = knn.predict(test_features)
    elif classifier=='MLP':
        predictions = train_MLP(train_features, train_labels, test_features)
    elif classifier=='Proto':
        classifier = ProtoNetClassifier(n_classes=15)
        classifier.compute_prototypes(train_features, train_labels)
        predictions = classifier.predict(test_features)
    
    # 在测试集上预测并评估
    accuracy = accuracy_score(test_labels, predictions)
    msg = "ResNet" + str(model_ver) + " " + str(classifier) + " Test Accuracy:" + str(accuracy);
    print(msg)
    
    # 输出 classification_report
    print("\n📊 Classification Report:")
    print(classification_report(test_labels, predictions, digits=4))
    
    # 输出混淆矩阵
    print("🔍 Confusion Matrix:")
    print(confusion_matrix(test_labels, predictions))
