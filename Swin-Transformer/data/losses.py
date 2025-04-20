import torch
import torch.nn as nn
from pytorch_toolbelt.losses import FocalLoss
from timm.loss import cross_entropy

class FocalCrossEntropyCombo(nn.Module):
    def __init__(self,
                 ce_ratio=0.5,
                 focal_weight=0.5,
                 ce_weight=None,
                 gamma=2.0,
                 reduction='mean'):
        super().__init__()
        self.ce_ratio = ce_ratio
        self.focal_ratio = 1.0 - ce_ratio
        self.gamma = gamma
        self.reduction = reduction

        self.ce_weight = None
        if ce_weight is not None:
            self.ce_weight = ce_weight.clone().detach()

        self.focal = FocalLoss(alpha=None, gamma=self.gamma, reduction=self.reduction)

    def forward(self, inputs, targets):
        # 自动适配 dtype / device 以避免 AMP 类型不一致
        if self.ce_weight is not None:
            self.ce_weight = self.ce_weight.to(dtype=inputs.dtype, device=inputs.device)

        ce_loss = cross_entropy(inputs, targets, weight=self.ce_weight, reduction=self.reduction)
        focal_loss = self.focal(inputs, targets)

        total_loss = self.ce_ratio * ce_loss + self.focal_ratio * focal_loss

        # debug 打印
        if not torch.isnan(total_loss).any():
            print(f"[Loss Debug] CE: {ce_loss.item():.4f} | Focal: {focal_loss.item():.4f} | Total: {total_loss.item():.4f}")
        else:
            print(f"[Loss Debug] CE: nan | Focal: nan | Total: nan")

        return total_loss
