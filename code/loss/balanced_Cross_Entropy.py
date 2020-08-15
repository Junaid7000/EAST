import torch
from torch import nn


#TODO: add weight factor for given dataset
class ClassBalanceCrossEntropyLoss(nn.Module):
    def __init__(self, weight = 1):
        super(ClassBalanceCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_ground):
        loss = self.cross_entropy_loss(y_pred, y_ground)
        return loss





