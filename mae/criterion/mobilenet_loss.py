# -*- coding: UTF-8 -*-
'''
@Author  ：LQR
@Project ：bird_sed_c_3 
@File    ：mobilenet_loss.py
@IDE     ：PyCharm 
@Date    ：2021/7/2 17:33 
'''
import torch
from torch import nn


class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * \
            (1. - probas)**self.gamma * bce_loss + \
            (1. - targets) * probas**self.gamma * bce_loss
        loss = loss.mean()
        return loss