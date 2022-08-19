import torch.nn as nn
import torch

class FingerLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
    def forward(self, gesture, landmarks,gesture_gt, landmark_gt):
        
        category_loss = self.entropy(gesture, gesture_gt)
        mse_loss = nn.functional.mse_loss(landmarks,landmark_gt)

        return mse_loss*category_loss