import torch
from torch import nn


class BinaryDiceLoss:
    """
    Dice loss implementation for two class segmentation
    """
    def __init__(self, eps=1e-8):
        """
        :param eps: Smooths denominator
        """
        self.eps = eps

    def __call__(self, y_true, logits):
        """
        :param y_true: Groundtruth of shape Batch x 1 x W x H
        :param logits: Model logits if shape Batch x 1 x W x H
        :return: Loss value
        """
        y_pred = nn.Sigmoid()(logits)
        num = 2 * torch.sum(torch.mul(y_true, y_pred))
        den = torch.sum(y_pred ** 2) + torch.sum(y_true ** 2) + self.eps
        res = 1 - num / den
        return res


class BCELoss:
    """
    Binary cross entropy loss function. Simplified version without numerical instability
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    """
    def __call__(self, y_true, logits):
        """
        :param y_true: Groundtruth of shape Batch x 1 x W x H
        :param logits: Model logits if shape Batch x 1 x W x H
        :return: Loss value
        """
        return (nn.ReLU()(logits) - y_true * logits + torch.log(1 + torch.exp(-torch.abs(logits)))).mean()


class BinaryTverskyLoss:
    """
    Focal Tversky loss from the article: https://arxiv.org/abs/1706.05721. Should deal with class imbalance
    """
    def __init__(self, beta=0.6, gamma=1.0, eps=1e-8):
        """
        :param beta: Weight parameter, Dice loss is obtained if equals 0.5
        :param gamma: Focal parameter
        :param eps: Smooths denominator
        """
        self.beta = beta
        self.eps = eps
        self.gamma = gamma

    def __call__(self, y_true, logits):
        """
        :param y_true: Groundtruth of shape Batch x 1 x W x H
        :param logits: Model logits if shape Batch x 1 x W x H
        :return: Loss value
        """
        y_pred = nn.Sigmoid()(logits)

        true_pos = torch.sum(torch.mul(y_true, y_pred))
        false_pos = torch.sum(torch.mul(1 - y_true, y_pred))
        false_neg = torch.sum(torch.mul(y_true, 1 - y_pred))

        ti = true_pos / (true_pos + (1 - self.beta) * false_pos + self.beta * false_neg + self.eps)
        tl = (1 - ti)**self.gamma

        return tl
