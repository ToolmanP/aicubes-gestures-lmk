import torch.nn as nn
import torch

class ConvBlock(nn.Module):

    def __init__(self,in_channels:int,out_channels:int) -> None:
        super().__init__()
        
        middle_channels = int(out_channels/2)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels,middle_channels,kernel_size=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(True),
            nn.Conv2d(middle_channels,middle_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(True),
            nn.Conv2d(middle_channels,out_channels,kernel_size=1)
        )

    def forward(self,x:torch.Tensor):
        return self.conv(x)

class SkipBlock(nn.Module):
    
    def __init__(self,in_channels:int,out_channels:int) -> None:
        super().__init__()
        
        if in_channels == out_channels:
            self.conv = nn.Identity();
        else:
            self.conv = nn.Conv2d(in_channels,out_channels,1)
                
    def forward(self,x:torch.Tensor):
        return self.conv(x)

class ResidualBlock(nn.Module):

    def __init__(self,in_channels:int,out_channels:int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels,out_channels)
        self.skip = SkipBlock(in_channels,out_channels)
    
    def forward(self,x:torch.Tensor):
        return self.conv(x)+self.skip(x)

class HourGlass(nn.Module):

    def __init__(self,in_channels:int,out_channels:int,levels:int) -> None:
        super().__init__()
        self.levels = levels
        
        self.skips = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for _ in range(self.levels):
            self.skips.append(SkipBlock(in_channels,out_channels))
            self.downs.append(nn.Sequential(
                nn.MaxPool2d(2,2),
                ResidualBlock(in_channels,out_channels),
            ))
            self.ups.append(nn.Sequential(
                ResidualBlock(in_channels,out_channels),
                nn.UpsamplingBilinear2d(scale_factor=2)
            ))
        
        self.middle = ResidualBlock(in_channels,out_channels)

    def forward(self,x:torch.Tensor):
        skip_history = list()
        
        for skip,down in zip(self.skips,self.downs):
            skip_history.append(skip(x))
            x = down(x)
            
        x = self.middle(x)
        
        for up,skipx in zip(self.ups,skip_history[::-1]):
            x = up(x)
            x = x + skipx
        
        return x