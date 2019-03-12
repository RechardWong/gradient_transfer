# -*- coding:utf-8 -*-
"""
Creator: hlc
Date: 2019/3/9
Time: 16:10

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class RegressionLoss(nn.Module):

    def __init__(self):
        super(RegressionLoss, self).__init__()

    def forward(self, embed1, embed2, size_average=True):
        diff = (embed1 - embed2)
        dist = diff.view(diff.size(0), -1).pow(2).sum(1).pow(.5)
        return dist.mean() if size_average else dist.sum()
