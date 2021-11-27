from PIL import Image
import os
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import utile


class RoadDataset(Dataset):
    def __init__(self, image_path, mask_path, transform=None, crop=False, crop_size=224, stride=16, padding=0):
        self.transform = transform
        self.images = self.load_images(image_path)
        self.masks = self.load_images(mask_path)
        if crop:
            self.images = utile.img_crop(self.images, crop_size, stride, padding)
            self.masks =utile.img_crop(self.masks, crop_size, stride, padding)

    def get_images(self):
        return self.images, self.masks

    @staticmethod
    def load_images(image_path):
        images = []
        for img in os.listdir(image_path):
            path = os.path.join(image_path, img)
            image = Image.open(path)
            images.append(np.asarray(image))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(self.images[index]/255, dtype=np.float32)
        mask = np.asarray(self.masks[index], dtype=np.float32)
        mask = (mask > 100).astype(float)

        if self.transform is not None:
            output = self.transform(image=image, mask=mask)
            image = output['image']
            mask = output['mask']

        return image, mask


if __name__ == '__main__':
    tr=ToTensorV2()
    a=np.zeros((256,256,3))
    a[:,:,1]=1
    a[:,:,2]=2
    b=tr(image=a)['image']
    print(b.shape)
    print(b[0])
    print(b[1])
    print(b[2])
