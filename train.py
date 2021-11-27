import os

import torchvision.models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from model import *
from utile import get_loader, accuracy, img_crop
from albumentations import *
from Losses import *

Train_image_path = 'data/training/images'
Train_mask_path = 'data/training/groundtruth'
Test_image_path = 'data/testing/images'
Test_mask_path = 'data/testing/groundtruth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Learning_rate = 1e-4
Batch_size = 4  # 4
Num_epochs = 3000
Num_workers = 0
Image_size = 400
Pin_memory = True
Pretrained = False
freeze_epochs = 10


def train_model(model, loader, optimizer, criterion, scaler):
    acc_loss = 0
    for i, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.unsqueeze(1).to(device)

        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)
            acc_loss += loss.item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return acc_loss


def main(m):
    train_transform = Compose(
        [
            Flip(),
            Transpose(),
            Rotate(),
            ToTensorV2()
        ]
    )

    net = LinkNet(in_channel=3, out_channel=1,resnet=m(pretrained=True),
                  filters=[256, 512, 1024, 2048]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=Learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    loader = get_loader(Train_image_path, Train_mask_path, train_transform, Batch_size, Num_workers, True, Pin_memory,
                        crop=False, crop_size=Image_size, stride=16, padding=0)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    test_transform = Compose(
        [
            Flip(),
            Transpose(),
            Rotate(),
            ToTensorV2()
        ]
    )
    test_loader = get_loader(Test_image_path, Test_mask_path, test_transform, Batch_size, Num_workers, False,
                             Pin_memory, crop=False, crop_size=Image_size, stride=16, padding=0)

    logs = []
    loop = tqdm(range(Num_epochs))
    max_f1 = 0
    for e in loop:
        if Pretrained:
            net.conv1.requires_grad_(e > freeze_epochs)
            net.bn1.requires_grad_(e > freeze_epochs)
            net.maxpool1.requires_grad_(e > freeze_epochs)
            net.encoders.requires_grad_(e > freeze_epochs)
        loss = train_model(net, loader, optimizer, criterion, scaler)
        # scheduler.step()
        log = accuracy(test_loader, net, device)
        log['epochs'] = e
        log['loss'] = loss
        logs.append(log)
        if log['f1 score'].item() > max_f1:
            max_f1 = log['f1 score'].item()
            if max_f1 > 80.0:
                torch.save(net, 'Checkpoints/'+m.__name__ + '_max_f1.pth')
        loop.set_postfix(loss=loss, acc=log['acc'].item(), dice_score=log['dice score'].item(),
                         f1_score=log['f1 score'].item(), max_f1=max_f1)

    f = open('logs/LinkNet_'+m.__name__+'_results', mode="w")
    for log in logs:
        f.write(str(log) + '\n')
    f.close()

    torch.save(net, 'Checkpoints/LinkNet_'+m.__name__ + '.pth')


if __name__ == '__main__':
    model=[torchvision.models.resnet50,torchvision.models.resnet101,torchvision.models.resnet152,
           torchvision.models.resnext50_32x4d, torchvision.models.resnext101_32x8d]
    for m in model:
        print(m.__name__)
        main(m)
