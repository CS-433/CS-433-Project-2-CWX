import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice Loss"""
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        num = predict.size(0)
        predict = torch.sigmoid(predict)
        pre = predict.view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score


class BCEDiceLoss(nn.Module):
    """Combination of Dice Loss and BCE Loss"""
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predict, target):
        bce_loss = self.bce(predict, target)
        dice_loss = self.dice(predict, target)
        return dice_loss + bce_loss


class focal_loss(nn.Module):
    """Focal Loss"""

    def __init__(self, gamma=2., alpha=.25, eps=1e-7):
        super(focal_loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        gamma = self.gamma
        alpha = self.alpha
        eps = self.eps

        y_pred_1 = torch.where(torch.gt(y_pred, eps), y_pred, eps * torch.ones_like(y_pred))
        y_pred_2 = torch.where(torch.le(y_pred_1, 1. - eps), y_pred_1, (1. - eps) * torch.ones_like(y_pred_1))
        pt_1 = torch.where(torch.eq(y_true, 1), y_pred_2, torch.ones_like(y_pred_2))
        pt_0 = torch.where(torch.eq(y_true, 0), y_pred_2, torch.zeros_like(y_pred_2))
        loss = -torch.sum(alpha * torch.pow(1. - pt_1, gamma) * torch.log(pt_1)) - torch.sum(
            (1 - alpha) * torch.pow(pt_0, gamma) * torch.log(1. - pt_0))
        return loss


class mix_loss(nn.Module):
    """Combination of Focal Loss and Dice Loss"""
    def __init__(self, rate=1e5, gamma=2., alpha=.25, eps=1e-7):
        super().__init__()
        self.rate = rate
        self.focal_loss = focal_loss(gamma, alpha, eps)
        self.dice_coef = DiceLoss()

    def forward(self, y_pred, y_true):
        f_loss = self.focal_loss(y_pred, y_true)
        d = self.dice_coef(y_pred, y_true)
        return f_loss + self.rate * d
