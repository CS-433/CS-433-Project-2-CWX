import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        num = predict.size(0)
        predict = torch.sigmoid(predict)
        pre = predict.view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predict, target):
        bce_loss = self.bce(predict, target)
        dice_loss=self.dice(predict,target)
        return dice_loss + bce_loss