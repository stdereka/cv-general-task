import torch
from torch import nn


class BinaryDiceLoss:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, y_true, logits):
        y_pred = nn.Sigmoid()(logits)
        num = 2 * torch.sum(torch.mul(y_true, y_pred))
        den = torch.sum(y_pred ** 2) + torch.sum(y_true ** 2) + self.eps
        res = 1 - num / den
        return res


class BCELoss:
    def __call__(self, y_true, logits):
        return (nn.ReLU()(logits) - y_true * logits + torch.log(1 + torch.exp(-torch.abs(logits)))).mean()


class BinaryTverskyLoss:
    def __init__(self, beta=0.6, eps=1e-8):
        self.beta = beta
        self.eps = eps

    def __call__(self, y_true, logits):
        y_pred = nn.Sigmoid()(logits)
        num = torch.sum(torch.mul(y_true, y_pred))
        den = torch.sum(torch.mul(y_true, y_pred) + torch.mul(y_true, (1 - self.beta) * (1 - y_pred)) +
                        torch.mul(self.beta * (1 - y_true), y_pred)) + self.eps
        res = 1 - num / den
        return res
