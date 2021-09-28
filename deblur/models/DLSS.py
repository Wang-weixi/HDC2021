import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import * 
class DLSS(nn.Module):
    def __init__(self,channels,):
        super(DLSS, self).__init__()
        layers1 = []
        layers1.append(nn.Upsample(scale_factor=2, mode='bicubic', align_corners=None))
        self.net1 = nn.Sequential(*layers1)
        layers2 = []
        layers2.append(nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=3, padding=1, bias=True))
        layers2.append(nn.ReLU(inplace=True))
        layers2.append(nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1, bias=True))
        layers2.append(nn.Upsample(scale_factor=2, mode='bicubic', align_corners=None))
        layers2.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True))
        layers2.append(nn.ReLU(inplace=True))
        layers2.append(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=True))
        self.net2 = nn.Sequential(*layers2)
    def forward(self, input):
        return self.net1(input)+self.net2(input)