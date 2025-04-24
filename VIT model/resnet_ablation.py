import torch
import timm
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F


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


def load_model():

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


class ResidualAblatedBlock(nn.Module):
    def __init__(self, original_block):

        super().__init__()
     
        self.norm1 = original_block.norm1
        self.attn = original_block.attn
        self.drop_path = original_block.drop_path  # 如果 drop_path_rate 为 0 则为 Identity
        self.norm2 = original_block.norm2
        self.mlp = original_block.mlp

    def forward(self, x):
 
        x1 = self.attn(self.norm1(x))
        x1 = self.drop_path(x1)

        x = x1
        x2 = self.mlp(self.norm2(x))
        x2 = self.drop_path(x2)

        x = x2
        return x

def ablate_residual_range(model, start_index, end_index):


    if hasattr(model, 'blocks'):
        for i in range(start_index, end_index):
            print(f"消融 Transformer block {i} 的残差连接")
            model.blocks[i] = ResidualAblatedBlock(model.blocks[i])
    else:
        print("模型没有 'blocks' 属性，无法进行残差连接消融")
    return model


def evaluate_model(model, data_loader):

    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total


if __name__ == '__main__':
    
    test_dataset = datasets.ImageFolder(root=test_folder, transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    

    model_full = load_model()
    acc_full = evaluate_model(model_full, test_loader)
    print("=== 完整模型 ===")
    print("完整模型准确率: {:.2%}".format(acc_full))
    
   
    num_blocks = len(model_full.blocks)
    print("Transformer 层总数：", num_blocks)
    
  
    groups = [(0, 1), (7, 8), (11, 12)]
    
    for group_id, (start, end) in enumerate(groups, 1):
        print(f"\n=== 消融实验：组 {group_id}（Transformer blocks {start} 到 {end-1} 的残差连接） ===")
        model_ablate = load_model()  
        model_ablate = ablate_residual_range(model_ablate, start, end)
        acc_ablate = evaluate_model(model_ablate, test_loader)
        print("禁用组 {} 内残差连接后的模型准确率: {:.2%}".format(group_id, acc_ablate))