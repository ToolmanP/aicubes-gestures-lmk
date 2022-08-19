from .const import *
from .transform import normalize

import numpy as np
import cv2, os

from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur
from torch.utils.data import Dataset
import torch


class FingerDataset(Dataset):

    def __init__(self,
                file_list:str,
                path:str,
                ):
        
        self.path = path
        self.shape = INSHAPE
        self.transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(INSHAPE)
            ]
        )

        with open(file_list, 'r') as f:
            self.lines = f.readlines()
    
    def __getitem__(self, index:int) -> torch.Tensor:

        img:torch.Tensor
        heatmap:torch.Tensor
        landmark:np.ndarray
        category:int

        line = self.lines[index].strip().split()
        x1, y1, x2, y2 = np.asarray(line[2:6], dtype=np.int32)
        img_path = os.path.join(self.path, line[0])

        img = cv2.imread(img_path)[y1:y2, x1:x2, :]
        img = normalize(img)
        img = self.transforms(img)

        landmark = np.asarray(line[6:90], dtype=np.float32)

        category = int(line[1])
        
        heatmap = torch.zeros([NLANDMARKS,*INSHAPE])
        
        for ch,mark in enumerate(landmark.reshape([-1,2])):
            x,y = mark
            heatmap[ch,int(x),int(y)]=1
        
        heatmap = gaussian_blur(heatmap,7,1.5)
        
        return (img, landmark, heatmap, category)

    def __len__(self) -> int:
        return len(self.lines)