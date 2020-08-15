import torch
from torch import nn


class IOULoss(nn.Module):
    def __init__(self):
        super(IOULoss, self).__init__()
        pass

    def forward(self, y_pred, y_truth):

        return self.calculate_iou(y_pred, y_truth)

    def calculate_iou(self, y_pred, y_truth):

        h_pred = y_pred[:, :, 2] - y_pred[:, :, 0] 
        w_pred = y_pred[:, :, 3] - y_pred[:, :, 1]

        h_truth = y_truth[:, :, 2] - y_truth[:, :, 0] 
        w_truth = y_truth[:, :, 3] - y_truth[:, :, 1]

        h_inter = y_truth[:, :, 2] - y_pred[:, :, 2] #confirm why adding 1 is important
        w_inter = y_truth[:, :, 3] - y_pred[:, :, 1]

        area_pred = torch.abs(h_pred*w_pred)
        area_truth = torch.abs(h_truth*w_truth)
        area_inter = torch.abs(h_inter*w_inter)

        iou = area_inter/(area_pred + area_truth + area_inter)

        return iou


class RotationLoss(nn.Module):
    def __init__(self):
        super(RotationLoss, self).__init__()
        pass

    def forward(self, y_pred, y_truth):
        
        cos_diff = torch.cos(y_pred, y_truth)

        return 1-cos_diff

    
