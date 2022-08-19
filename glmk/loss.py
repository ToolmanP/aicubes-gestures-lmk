import torch.nn as nn
import torch.nn.functional as f
import torch

class FingerLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, gesture, landmarks, heatmaps, gesture_gt, landmarks_gt, heatmap_gt):
        
        category_loss = self.entropy(gesture, gesture_gt)
        landmark_loss = f.mse_loss(landmarks,landmarks_gt)
        heatmap_loss = sum([f.mse_loss(heatmap,heatmap_gt) for heatmap in heatmaps])/len(heatmaps)
        return (0.5*landmark_loss+0.5*heatmap_loss)*category_loss