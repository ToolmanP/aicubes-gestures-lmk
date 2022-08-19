from .layers import ResidualBlock,SkipBlock,HourGlass

import torch.nn as nn
import torch

class HGN(nn.Module):

    def __init__(self,
                in_channels:int,
                out_channels:int,
                levels:int) -> None:
        super().__init__()
        
        self.hg = HourGlass(in_channels,in_channels,levels)

        self.conv = nn.Sequential(
            ResidualBlock(in_channels,in_channels),
            nn.Conv2d(in_channels,in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )
        self.out = nn.Conv2d(in_channels,out_channels)
        self.skip1 = SkipBlock(in_channels,in_channels)
        self.skip2 = SkipBlock(out_channels,in_channels)

    def forward(self,x:torch.Tensor):
        x1 = self.hg(x)
        x1 = self.conv(x1)
        out = self.out(x1)
        x1 = self.skip1(x1)
        x2 = self.skip2(out)
        return out,x1+x2+x

class SHGN(nn.Module):

    def __init__(self,
                in_channels:int,
                out_channels:int,
                levels:int,
                recursions:int) -> None:

        super().__init__()
        self.hgns = nn.ModuleList()
        for _ in range(recursions):
            self.hgns.append(HGN(in_channels,out_channels,levels))
    
    def forward(self,features:torch.Tensor):
        outs = []
        for hgn in self.hgns:
            out,features = hgn(features)
            outs.append(out)
        return outs

class SHGNClassifier(nn.Module):
    pass