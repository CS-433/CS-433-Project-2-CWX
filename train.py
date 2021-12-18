import os
import torch.random
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from model import *
from utile import get_loader, accuracy
from albumentations import *
from Losses import *

Train_image_path = 'data/training/images'
Train_mask_path = 'data/training/groundtruth'
Test_image_path = 'data/testing/images'
Test_mask_path = 'data/testing/groundtruth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Learning_rate = 1e-4
Batch_size = 4  # 4
Num_epochs = 1000
Num_workers = 0
Image_size = 400
Pin_memory = True
one_hot = False
crop = False
crop_size = 224
stride = 16
padding = 0
T = 50

torch.manual_seed(0)
torch.cuda.manual_seed(0)


def train_model(model, loader, optimizer, criterion, scaler):
    """Train the model"""
    acc_loss = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = 0
            for i in range(output.shape[1]):
                pred = output[:, i, :, :].unsqueeze(1)
                loss += criterion(pred, target)

            acc_loss += loss.item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return acc_loss


def main(name):
    # name = 'final'
    print(name)
    if one_hot:
        name += "_with_one_hot"

    # Create the network
    net = LinkNet(in_channel=3, out_channel=1, resnet=torchvision.models.resnet152(pretrained=True),
                  filters=[256, 512, 1024, 2048]).to(device)
    # Define the criterion and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=Learning_rate)

    # Define the scheduler and scaler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)
    scaler = torch.cuda.amp.GradScaler()

    # Define the data augmentation used for the training set, and also create the data loader for it
    train_transform = Compose(
        [
            Resize(height=Image_size, width=Image_size),
            Flip(),
            Transpose(),
            Rotate(),
            CoarseDropout(max_holes=8, max_height=8, max_width=8),
            OneOf([
                OpticalDistortion(),
                GridDistortion(),
                ElasticTransform()
            ], p=0.5),
            ToTensorV2(),

        ]
    )

    train_loader = get_loader(Train_image_path, Train_mask_path, train_transform, Batch_size, Num_workers, True,
                              Pin_memory, one_hot=one_hot, crop=crop, crop_size=crop_size, stride=stride,
                              padding=padding)

    # Define the data augmentation used for the validation set, and also create the data loader for it
    val_transform = Compose(
        [
            Resize(height=Image_size, width=Image_size),
            Flip(),
            Transpose(),
            ToTensorV2()
        ]
    )
    val_loader = get_loader(Test_image_path, Test_mask_path, val_transform, Batch_size, Num_workers, False,
                            Pin_memory, rotations=[0, 90, 180, 270],
                            one_hot=one_hot, crop=crop, crop_size=crop_size, stride=stride, padding=padding)

    # Train the model, then save the training logs and the best model
    logs = []
    loop = tqdm(range(Num_epochs))
    max_f1 = 0

    for e in loop:
        loss = train_model(net, train_loader, optimizer, criterion, scaler)
        scheduler.step()
        log = accuracy(val_loader, net, device, one_hot=one_hot)
        log['epochs'] = e
        log['loss'] = loss
        logs.append(log)
        if log['f1 score'] > max_f1:
            max_f1 = log['f1 score']
            if max_f1 > 80.0:
                torch.save(net, 'Checkpoints/' + name + '_max_f1.pth')
        loop.set_postfix(loss=loss, acc=log['acc'], f1_score=log['f1 score'], max_f1=max_f1)

    # Save the logs into a file
    f = open('logs/' + name + '_results', mode="w")
    for log in logs:
        f.write(str(log) + '\n')
    f.close()
    # torch.save(net, 'Checkpoints/' + name + '.pth')
    return net


if __name__ == '__main__':
    if not os.path.exists('Checkpoints'):
        os.makedirs('Checkpoints')
    if not os.path.exists('logs'):
        os.makedirs('logs')

    name = 'final'
    main(name)
