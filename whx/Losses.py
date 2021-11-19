import torch
import torch.nn as nn

class focal_loss(nn.Module):
    """focal loss"""  
    def __init__(self, gamma=2., alpha=.25, eps=1e-7):
        super(focal_loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
    def forward(self, y_pred, y_true):
        gamma = self.gamma
        alpha = self.alpha
        eps = self.eps

        y_pred_1 = torch.where(torch.gt(y_pred, eps), y_pred, eps*torch.ones_like(y_pred))
        y_pred_2 = torch.where(torch.le(y_pred_1, 1.-eps), y_pred_1, (1.-eps)*torch.ones_like(y_pred_1))
        pt_1 = torch.where(torch.eq(y_true, 1), y_pred_2, torch.ones_like(y_pred_2))
        pt_0 = torch.where(torch.eq(y_true, 0), y_pred_2, torch.zeros_like(y_pred_2))
        loss = -torch.sum(alpha*torch.pow(1.-pt_1, gamma)*torch.log(pt_1))-torch.sum((1-alpha)*torch.pow(pt_0, gamma)*torch.log(1.-pt_0))
        return loss

class dice_coef(nn.Module):
    """dice"""
    def __init__(self):
        super(dice_coef, self).__init__()
    def forward(self, y_pred, y_true):
        if isinstance(y_pred, (list, tuple)):
            y_pred = y_pred[0]
        # torch.view(-1) ：把y_true和y_pred变成一行
        y_true_f = y_true.view(-1).type(torch.float)
        y_pred_f = y_pred.view(-1).type(torch.float)
        mul = torch.mul(y_true_f, y_pred_f)
        intersection = torch.sum(mul)
        return (2. * intersection + 1) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + 1)

class mix_loss(nn.Module):
    def __init__(self, rate=1e4, gamma=2., alpha=.25, eps=1e-7):
        super().__init__()
        self.rate = rate
        self.focal_loss = focal_loss(gamma, alpha, eps)
        self.dice_coef = dice_coef()
    def forward(self, y_true, y_pred):
        f_loss = self.focal_loss(y_true, y_pred)
        d = self.dice_coef(y_true, y_pred)
        d_loss = 1. - d
        return f_loss + self.rate * d_loss