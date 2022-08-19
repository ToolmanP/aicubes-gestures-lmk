from .transform import normalize
from .const import *

import numpy as np
import cv2, os

from torchvision import transforms
from torch.utils.data import Dataset
import torch


class FingerDataset(Dataset):

    def __init__(self, file_list, path):
        
        self.path = path
        self.transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(SIZE)
            ]
        )
        with open(file_list, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, index):

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
        
        landmark = landmark.reshape([2,-1])
        
        return (img, landmark, category)

    def __len__(self):
        return len(self.lines)