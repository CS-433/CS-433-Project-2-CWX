import torch
from PIL import Image
import os
from torch.utils.data import Dataset
import _thread
import utile
from albumentations import *


class RoadDataset(Dataset):
    """The class RoadDataset loads the data and executes the pre-processing operations on it"""
    def __init__(self, image_path, mask_path, transform=None, one_hot=False, rotations=None,
                 crop=False, crop_size=224, stride=16, padding=0):
        self.transform = transform
        self.one_hot = one_hot
        self.images = self.load_images(image_path)
        self.masks = self.load_images(mask_path)
        self.images_augmented = []
        self.masks_augmented = []

        if rotations is not None:
            self.images, self.masks = self.rotate(self.images, self.masks, rotations)

        # Crop the images into patches with respect to the given size
        if crop:
            self.images = utile.img_crop(self.images, crop_size, stride, padding)
            self.masks = utile.img_crop(self.masks, crop_size, stride, padding)

        # Data augmentation
        for i in range(len(self.images)):
            output = self.transform(image=self.images[i], mask=self.masks[i])
            self.images_augmented.append(output['image'])
            self.masks_augmented.append(output['mask'])

    @staticmethod
    def rotate(images, masks, rotations):
        """This method applies rotations to the image according to the given angles"""
        ims = []
        msks = []
        for im, msk in zip(images, masks):
            for rotation in rotations:
                ims.append(rotate(im, rotation))
                msks.append(rotate(msk, rotation))
        return np.asarray(ims), np.asarray(msks)

    def get_images(self):
        return self.images, self.masks

    @staticmethod
    def load_images(image_path):
        """This method loads the images from the given path"""
        images = []
        for img in os.listdir(image_path):
            path = os.path.join(image_path, img)
            image = Image.open(path)
            images.append(np.asarray(image))
        return np.asarray(images)

    def augment(self, index):
        """This method applies data augmentation to the images"""
        output = self.transform(image=self.images[index], mask=self.masks[index])
        self.images_augmented[index] = output['image']
        self.masks_augmented[index] = output['mask']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """This method returns the image at a certain position and its mask"""
        image = self.images_augmented[index]
        mask = self.masks_augmented[index]
        """if self.transform is not None:
            output = self.transform(image=image, mask=mask)
            image = output['image']
            mask = output['mask']"""

        _thread.start_new_thread(self.augment, (index,))
        mask = mask.reshape((1,) + mask.shape)
        if self.one_hot:
            one_hot_mask = torch.zeros((2,) + mask.shape[1:])
            one_hot_mask.scatter_(0, mask.long(), 1).float()
            mask = one_hot_mask
        return (image/255), (mask > 100).float()


if __name__ == '__main__':
    pass
