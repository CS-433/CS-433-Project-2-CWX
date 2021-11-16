import torch
from dataset import RoadDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from PIL import Image
from helper import concatenate_images


def save_checkpoint(state, name='checkpoint'):
    torch.save(state, name)


def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint['state_dict'])


def get_loader(
        data_path,
        mask_path,
        transform,
        batch_size,
        num_walker,
        shuffle,
        pin_memory
):
    dataset = RoadDataset(data_path, mask_path, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_walker
    )
    return loader


def f1(pred, label):
    if pred.ndim == 4:
        pred = F.avg_pool2d(pred, kernel_size=16)
        pred[pred < 0.25] = 0
        pred[pred >= 0.25] = 1
    if label.ndim == 3:
        label = label.reshape((label.shape[0], 1, label.shape[1], label.shape[2]))
    if label.ndim == 4:
        label = F.avg_pool2d(label, kernel_size=16)
        label[label < 0.25] = 0
        label[label >= 0.25] = 1

    pred = pred.view(-1)
    label = label.view(-1)
    tp = (label * pred).sum().to(torch.float32)
    tn = ((1 - label) * (1 - pred)).sum().to(torch.float32)
    fp = ((1 - label) * pred).sum().to(torch.float32)
    fn = (label * (1 - pred)).sum().to(torch.float32)

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return 2 * precision * recall / (precision + recall + 1e-7)


def accuracy(loader, model, device, save_im=False, print_res=False):
    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    f1_score = 0
    log = dict()
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.unsqueeze(1).to(device)
            pred = torch.sigmoid(model(x))
            pred = (pred > 0.5).float()
            num_correct += (pred == y).sum()
            num_pixels += torch.numel(pred)
            dice_score += (2 * pred * y).sum() / ((pred + y).sum() + 1e-8)
            f1_score += f1(pred, y)
            if save_im:
                count = save_prediction(pred, y, count)

    if print_res:
        print(f"{num_correct}/{num_pixels}, acc {num_correct / num_pixels * 100:.2f}%, "
              f"dice score {dice_score / len(loader) * 100:.2f}%, "
              f"f1 score {f1_score / len(loader) * 100:.2f}%")

    log['acc'] = num_correct / num_pixels * 100
    log['dice score'] = dice_score / len(loader) * 100
    log['f1 score'] = f1_score / len(loader) * 100
    model.train()
    return log


def save_prediction(prediction, label, count=0):
    for p, l in zip(prediction, label):
        im = concatenate_images(p[0], l[0])
        im = Image.fromarray(im)
        im.save('output/test_' + str(count) + '.png')
        count += 1
    return count
