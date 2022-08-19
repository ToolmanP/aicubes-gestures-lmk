from telnetlib import X3PAD
from .layers import InvertedResidual as ConvBlock

import torch.nn as nn
import torch


class PFLD(nn.Module):
    def __init__(self,
                in_channels:int,
                out_channels:int) -> None:
        super().__init__()
        
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

        self.conv3 = nn.Sequential(
            ConvBlock(in_channels*2,32,stride=1,expand_ratio=2,residual=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
            )

        self.conv4 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.avg1 = nn.AvgPool2d(6)
        self.avg2 = nn.AvgPool2d(3)
        self.avg3 = nn.AvgPool2d(2)

        self.up = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels*2,in_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels,int(in_channels/2),kernel_size=3,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )

        self.fc = nn.Linear(224,out_channels*2)

    def forward(self,x:torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        features = self.up(x)
        x = self.conv3(x)
        x1 = self.avg1(x)
        x = self.conv4(x)
        x2 = self.avg2(x)
        x = self.conv5(x)
        x3 = self.avg3(x)

        x1 = x1.flatten(1)
        x2 = x2.flatten(1)
        x3 = x3.flatten(1)

        y = torch.cat([x1,x2,x3],dim=1)

        return self.fc(y),features