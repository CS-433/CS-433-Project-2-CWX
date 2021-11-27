import torch
from dataset import RoadDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from PIL import Image
from helper import concatenate_images
import albumentations as alb
import numpy as np
from albumentations.pytorch import ToTensorV2

from whx.utils import F1_coef, post_process


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
        pin_memory,
        crop=False,
        crop_size=224,
        stride=16,
        padding=0
):
    dataset = RoadDataset(data_path, mask_path, transform, crop=crop, crop_size=crop_size
                          , stride=stride, padding=padding)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_walker
    )
    return loader


def f1(pred, label):
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
            pred = (pred >= 0.5).float()
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


def different_prospective_prediction(image, model, rotations, transposes):
    predictions = []
    for rotation in rotations:
        im = alb.Rotate((rotation, rotation), p=1)(image=image)['image']
        im = ToTensorV2()(image=im)['image']
        prediction = torch.sigmoid(model(im.reshape(1, im.shape[0], im.shape[1], im.shape[2]).to('cuda')))
        prediction = prediction.cpu().detach().numpy()
        prediction = prediction.reshape(prediction.shape[-2], prediction.shape[-1])
        prediction = alb.Rotate((-rotation, -rotation), p=1)(image=prediction)['image']
        predictions.append(prediction)
        if transposes:
            im = alb.Rotate((rotation, rotation), p=1)(image=image)['image']
            im = alb.HorizontalFlip(p=1)(image=im)['image']
            im = ToTensorV2()(image=im)['image']
            prediction = torch.sigmoid(model(im.reshape(1, im.shape[0], im.shape[1], im.shape[2]).to('cuda')))
            prediction = prediction.cpu().detach().numpy()
            prediction = prediction.reshape(prediction.shape[-2], prediction.shape[-1])
            prediction = alb.HorizontalFlip(p=1)(image=prediction)['image']
            prediction = alb.Rotate((-rotation, -rotation), p=1)(image=prediction)['image']
            predictions.append(prediction)

    prediction = np.sum(predictions, axis=0) / len(predictions)
    prediction[prediction < 0.5] = 0
    prediction[prediction >= 0.5] = 1
    prediction = prediction * 255
    return prediction


def img_crop(images, size, stride=16, padding=0):
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
    return np.array(list_img_patches)


def combine_img(image, original_size, stride=16, padding=0):
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
