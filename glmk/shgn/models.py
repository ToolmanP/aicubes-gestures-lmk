import torch
import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self,x:torch.Tensor):
        pass

class HourGlass(nn.Module):

    def __init__(self,levels:int) -> None:
        super().__init__()
        self.levels = levels
    
    def forward(self,x:torch.Tensor):
        pass
