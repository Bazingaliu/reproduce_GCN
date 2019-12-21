import torch
import numpy as np
from torch import nn


class LabelSmoothLoss(nn.Module):
    """
    Cross entropy loss with label smooth.
    if sigma == 0.0, this loss equals to nn.CrossEntropyLoss
    """
    def __init__(self, weight: list = None, sigma: float = 0.0):
        super(LabelSmoothLoss, self).__init__()

        self.sigma = sigma
        self.distribution = None
        if weight is not None:
            self.distribution = torch.from_numpy(np.array(weight)).float()

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.cross_entropy_loss_one_hot = CrossEntropyLossOneHot()

    def forward(self, preds, labels):
        """
        preds: [batch_size, label_size]
        labels: [label_size]
        """
        batch_size = preds.shape[0]
        label_size = preds.shape[1]
        if self.distribution is None:
            uniform_dis = torch.ones([batch_size, label_size]) / label_size
        else:
            uniform_dis = self.distribution

        ce_loss = self.cross_entropy_loss(preds, labels)
        ud_loss = self.cross_entropy_loss_one_hot(preds, uniform_dis)
        ls_loss = (1.0 - self.sigma) * ce_loss + self.sigma * ud_loss
        return ls_loss


class CrossEntropyLossOneHot(nn.Module):
    def __init__(self):
        super(CrossEntropyLossOneHot, self).__init__()
        self.soft_max = nn.LogSoftmax(dim=-1)
        self.nll_loss = nn.NLLLoss()

    def forward(self, preds, labels):
        """
        preds: [batch_size, label_size]
        labels: [batch_size, label_size] - One hot encoding by ground truth
        """
        batch_size = preds.shape[0]

        soft_preds = self.soft_max(preds)
        mul_res = torch.mul(soft_preds, labels)
        sum_res = torch.sum(- mul_res, dim=-1)
        cross_entropy_loss = torch.sum(sum_res, dim=0) / batch_size

        return cross_entropy_loss


if __name__ == '__main__':
    preds = torch.from_numpy(np.array([[0.2, 0.8], [0.1, 0.9]])).float()
    labels = torch.from_numpy(np.array([1, 1])).long()

    loss = LabelSmoothLoss(sigma=0.1)
    a = loss(preds, labels)
    exit()
