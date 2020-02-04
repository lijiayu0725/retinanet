import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_l1_loss(pred, target, avg_factor=None, beta=0.11, weight=1):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return (loss * weight).sum() / avg_factor


class SmoothL1Loss(nn.Module):

    def __init__(self, beta=0.11):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, pred, target, avg_factor=None):
        loss_bbox = smooth_l1_loss(pred, target, beta=self.beta, avg_factor=avg_factor)
        return loss_bbox


# This method is only for debugging
def py_sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25, avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
    loss = (loss * weight).sum() / avg_factor
    return loss


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target, weight=None, avg_factor=None):
        loss_cls = self.loss_weight * py_sigmoid_focal_loss(pred, target, gamma=self.gamma, alpha=self.alpha,
                                                            avg_factor=avg_factor)
        return loss_cls
