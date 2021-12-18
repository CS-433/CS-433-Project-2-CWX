import torch
from dataset import RoadDataset
from torch.utils.data import DataLoader
import albumentations as alb
import numpy as np


def get_loader(
        data_path,
        mask_path,
        transform,
        batch_size,
        num_walker,
        shuffle,
        pin_memory,
        one_hot=False,
        rotations=None,
        crop=False,
        crop_size=224,
        stride=16,
        padding=0
):
    """Create the DataLoader class"""
    dataset = RoadDataset(data_path, mask_path, transform, one_hot=one_hot, rotations=rotations,
                          crop=crop, crop_size=crop_size, stride=stride, padding=padding)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_walker
    )
    return loader


def f1(pred, label):
    """Compute F1 score"""
    pred = pred.view(-1)
    label = label.view(-1)
    tp = (label * pred).sum().to(torch.float32)
    # tn = ((1 - label) * (1 - pred)).sum().to(torch.float32)
    fp = ((1 - label) * pred).sum().to(torch.float32)
    fn = (label * (1 - pred)).sum().to(torch.float32)
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps), precision, recall


def accuracy(loader, model, device, one_hot=False):
    """Compute the accuracy rate on the given dataset with the input model"""
    model.eval()
    log = dict()
    num_correct = 0
    num_pixels = 0
    f1_score = 0
    precision = 0
    recall = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            output = output[:, -1, :, :].unsqueeze(1)
            pred = torch.sigmoid(output)
            if one_hot:
                pred = pred.argmax(1)
                y = y.argmax(1)
            else:
                pred = (pred >= 0.5).float()
            num_correct += (pred == y).sum().item()
            num_pixels += torch.numel(pred)
            a, b, c = f1(pred, y)
            f1_score += a.item()
            precision += b.item()
            recall += c.item()

    log['acc'] = num_correct / num_pixels * 100
    log['f1 score'] = f1_score / len(loader) * 100
    log['precision'] = precision / len(loader) * 100
    log['recall'] = recall / len(loader) * 100
    model.train()
    return log


def create_different_prospective(images, rotations, transposes):
    """Apply transformations to the image and return different prospectives"""
    ims = []
    for image in images:
        for rotation in rotations:
            ims.append(alb.rotate(image, rotation))
            if transposes:
                im = alb.hflip(image)
                ims.append(alb.rotate(im, rotation))
    ims = np.array(ims)
    ims = torch.tensor(ims).transpose(1, -1).transpose(2, -1).float()
    return ims


def combine_different_prospective(images, rotations, transposes):
    """Combine predictions of different prospectives"""
    outputs = []
    index = 0
    while index < len(images):
        output = np.zeros(images[0].shape)
        for rotation in rotations:
            im = images[index, 0]
            output += alb.rotate(im, -rotation)
            index += 1
            if transposes:
                im = images[index, 0]
                im = alb.rotate(im, -rotation)
                output += alb.hflip(im)
                index += 1
        output = output / len(images)
        outputs.append(output)
    return np.array(outputs)


def img_crop(images, size, stride=16, padding=0):
    """Crop the image into patches of a given size"""
    list_img_patches = []
    for image in images:
        ndim = image.ndim
        imgwidth = image.shape[0] - size
        imgheight = image.shape[1] - size
        if ndim == 2:
            image = image.reshape((image.shape[0], image.shape[1], 1))
        new_image = np.zeros((image.shape[0] + 2 * padding, image.shape[1] + 2 * padding, image.shape[2]))
        new_image[padding:padding + image.shape[0], padding:padding + image.shape[1], :] = image
        image = new_image
        for i in range(padding, imgheight + padding + 1, stride):
            for j in range(padding, imgwidth + padding + 1, stride):
                im_patch = image[j - padding:j + size + padding, i - padding:i + size + padding, :]
                if ndim == 2:
                    im_patch = im_patch.reshape((im_patch.shape[0], im_patch.shape[1]))
                list_img_patches.append(im_patch)
    return np.array(list_img_patches, dtype=np.float32)


def combine_img(image, original_size, stride=16, padding=0):
    """Combine patches into the original image"""
    original_dim = image[0].ndim
    if image.ndim == 3:
        image = image.reshape(
            (image.shape[0], image.shape[1], image.shape[2], 1))
    img_size = int((1 + (original_size - image.shape[1] + 2 * padding) / stride) ** 2)
    res = []
    size = image.shape[1] - 2 * padding
    for i in range(0, image.shape[0], img_size):
        temp = image[i:i + img_size]
        org_img = np.zeros((original_size, original_size, temp.shape[-1]))
        counts = np.zeros((original_size, original_size, temp.shape[-1]))
        index = 0
        for j in range(0, original_size - size + 1, stride):
            for k in range(0, original_size - size + 1, stride):
                org_img[k:k + size, j:j + size] = temp[index, padding:temp.shape[1] - padding,
                                                  padding:temp.shape[2] - padding]
                counts[k:k + size, j:j + size] += 1
                index += 1
        if original_dim == 2:
            org_img = org_img.reshape((org_img.shape[0], org_img.shape[1]))
        res.append(np.uint8(org_img))
    return np.array(res)


def moving_average(net1, net2, alpha=1, loader=None):
    num1 = 0
    num2 = 0
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha
        num1 += torch.ones(param1.data.shape).sum()
        num2 += torch.ones(param2.data.shape).sum()
    print(accuracy(loader, net1, 'cuda', one_hot=False))
    print(accuracy(loader, net2, 'cuda', one_hot=False))
    print(num1,num2)
    return net1
