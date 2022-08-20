
from .models import *
from .const import *

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor

import torch.nn as nn
import torch

class Classifier(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
        self.down = nn.Sequential(
            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(12*12*512,4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True))
        
        self.fc2 = nn.Linear(4096,3)

        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(512,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

    def forward(self,features:torch.Tensor):
        features = self.down(features)
        gesture = self.fc1(features.flatten(1))
        gesture = self.fc2(gesture)
        gest_features = self.up(features)
        return gesture,gest_features

class CollabNetwork(nn.Module):

    def __init__(self,
                 shgn_levels: int,
                 shgn_iterations: int) -> None:
        super().__init__()

        self.shgn = SHGN(256,NLANDMARKS, levels=shgn_levels,
                         iterations=shgn_iterations)
        self.pfld = PFLD(256, NLANDMARKS)
        self.classifier = Classifier()

        self.conv_pfld = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.conv_shgn = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

    def forward(self, res: torch.Tensor, prev_gest:torch.Tensor, prev_pfld:torch.Tensor, prev_shgn:torch.Tensor) -> torch.Tensor:

        heatmaps,shgn_feature = self.shgn(res + prev_gest + prev_pfld)
        landmarks,pfld_feature = self.pfld(res + prev_gest + prev_shgn)

        joint_feature = self.conv_shgn(shgn_feature)+self.conv_pfld(pfld_feature)
        print(joint_feature.shape)
        print(res.shape)
        gesture,gest_feature= self.classifier(joint_feature+res)
        return gesture,landmarks,heatmaps,gest_feature,pfld_feature,shgn_feature
