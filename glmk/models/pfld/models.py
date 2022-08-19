from .layers import InvertedResidual as ConvBlock

import torch.nn as nn
import torch


class PFLD(nn.Module):
    def __init__(self,
                in_channels:int,
                out_channels:int,
                size:torch.Tensor) -> None:
        super().__init__()
        
        self.inconv = nn.Sequential(
            nn.Conv2d(in_channels )
        )
        self.conv1 = nn.Sequential(
            ConvBlock(in_channels,in_channels,stride=2,expand_ratio=2,residual=False),
            ConvBlock(in_channels,in_channels,stride=1,expand_ratio=2,residual=False),
            ConvBlock(in_channels,in_channels,stride=1,expand_ratio=2,residual=True),
            ConvBlock(in_channels,in_channels,stride=1,expand_ratio=2,residual=True),
            ConvBlock(in_channels,in_channels,stride=1,expand_ratio=2,residual=True),
            ConvBlock(in_channels,in_channels,stride=1,expand_ratio=2,residual=True),
        )

        self.conv2 = nn.Sequential(
            ConvBlock(in_channels,in_channels*2,stride=2,expand_ratio=2,residual=False),
            ConvBlock(in_channels*2,in_channels*2,stride=1,expand_ratio=4,residual=False),
            ConvBlock(in_channels*2,in_channels*2,stride=1,expand_ratio=4,residual=True),
            ConvBlock(in_channels*2,in_channels*2,stride=1,expand_ratio=4,residual=True),
            ConvBlock(in_channels*2,in_channels*2,stride=1,expand_ratio=4,residual=True),
            ConvBlock(in_channels*2,in_channels*2,stride=1,expand_ratio=4,residual=True)
        )

        self.conv3 = ConvBlock(in_channels*2,16,stride=1,expand_ratio=2,residual=False)
        self.conv4 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.conv5 = nn.Conv2d(32,128,kernel_size=6,stride=1,padding=0)

        self.avg1 = nn.AvgPool2d(12)
        self.avg2 = nn.AvgPool2d(6)
        
        self.fc = nn.Linear(1,out_channels*2)
    
    def forward(self,x:torch.Tensor):
        pass


class PFLDClassifier(nn.Module):
    pass
