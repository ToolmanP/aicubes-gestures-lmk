import torch.nn as nn
import torch

class SkipBlock(nn.Module):
    
    def __init__(self,in_channels:int,out_channels:int) -> None:
        super().__init__()
        
        if in_channels == out_channels:
            self.conv = nn.Identity();
        else:
            self.conv = nn.Conv2d(in_channels,out_channels,1)
                
    def forward(self,x:torch.Tensor):
        return self.conv(x)

class InvertedResidual(nn.Module):
    
    def __init__(self,
                in_channels:int,
                out_channels:int,
                stride:int,
                expand_ratio:int,
                residual:bool=False
                ) -> None:
        super().__init__()

        self.residual = residual
        middle_channels = in_channels*expand_ratio

        self.skip = SkipBlock(in_channels,out_channels)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,middle_channels,kernel_size=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(True),
            nn.Conv2d(middle_channels,middle_channels,kernel_size=3,stride=stride,padding=1,groups=middle_channels),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(True),
            nn.Conv2d(middle_channels,out_channels,kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self,x:torch.Tensor):
        if self.residual:
            return self.skip(x)+self.conv(x)
        else:
            return self.conv(x)