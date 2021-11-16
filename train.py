from tqdm import tqdm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from model import Unet
from utile import get_loader, accuracy
from albumentations import *

Train_image_path = 'data/training/images'
Train_mask_path = 'data/training/groundtruth'
Test_image_path = 'data/testing/images'
Test_mask_path = 'data/testing/groundtruth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Learning_rate = 1e-4
Batch_size = 4  # 64
Num_epochs = 500
Num_workers = 2
Image_size = 400
Pin_memory = True


def train_model(model, loader, optimizer, criterion, scaler):
    for i, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.unsqueeze(1).to(device)

        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def main():
    train_transform = Compose(
        [
            RandomSizedCrop(min_max_height=(128, 400), height=Image_size, width=Image_size),
            Flip(),
            Transpose(),
            Rotate(),
            OneOf([
                OpticalDistortion(),
                GridDistortion(),
                ElasticTransform()
            ], p=0.2),
            GaussNoise(var_limit=(0, 1e-5)),
            Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ToTensorV2()
        ]
    )

    model = Unet(in_channel=3, out_channel=1, nb_res_block=0, p=0).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    loader = get_loader(Train_image_path, Train_mask_path, train_transform, Batch_size, Num_workers, True, Pin_memory)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

    test_transform = Compose(
        [
            Resize(height=Image_size, width=Image_size),
            Flip(),
            Transpose(),
            RandomRotate90(),
            ShiftScaleRotate(),
            Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ToTensorV2()
        ]
    )
    test_loader = get_loader(Test_image_path, Test_mask_path, test_transform, Batch_size, Num_workers, False,
                             Pin_memory)

    logs = []
    loop = tqdm(range(Num_epochs))
    max_f1 = 0
    for e in loop:
        train_model(model, loader, optimizer, criterion, scaler)
        # scheduler.step()
        log = accuracy(test_loader, model, device)
        log['epochs'] = e
        logs.append(log)
        loop.set_postfix(acc=log['acc'].item(), dice_score=log['dice score'].item(), f1_score=log['f1 score'].item())
        if log['f1 score'].item() > max_f1:
            max_f1 = log['f1 score'].item()
            if e > 100:
                torch.save(model, 'Checkpoints/Unet_max_f1_1.pth')

    f = open('results', mode="w")
    for log in logs:
        f.write(str(log) + '\n')
    f.close()

    accuracy(test_loader, model, device)
    torch.save(model, 'Checkpoints/Unet_1.pth')


if __name__ == '__main__':
        main()
