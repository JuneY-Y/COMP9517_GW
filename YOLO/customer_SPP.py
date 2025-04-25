# customer spp.py
import torch
import torch.nn as nn

class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[5, 9, 13]):
        super(SPP, self).__init__()
        self.cv1 = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in pool_sizes
        ])
        self.cv2 = nn.Conv2d(out_channels * (len(pool_sizes) + 1), out_channels, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        outs = [x] + [pool(x) for pool in self.poolings]
        x = torch.cat(outs, dim=1)
        return self.cv2(x)