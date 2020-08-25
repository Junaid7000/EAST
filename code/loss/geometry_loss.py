import torch
from torch import nn


class IOULoss(nn.Module):
    def __init__(self):
        super(IOULoss, self).__init__()
        pass

    def forward(self, y_pred, y_truth):
        pass

    def calculate_iou(self, y_pred, y_truth):

        pass

class RotationLoss(nn.Module):
    def __init__(self):
        super(RotationLoss, self).__init__()
        pass

    def forward(self, y_pred, y_truth):

        cos_diff = torch.cos(y_pred, y_truth)
        
        return 1-cos_diff

    
