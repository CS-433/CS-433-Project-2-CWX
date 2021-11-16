from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class RoadDataset(Dataset):
    def __init__(self, image_path, mask_path, transform=None):
        self.image_path = image_path
        self.mask_path = mask_path
        self.transform = transform
        self.images = os.listdir(image_path)
        self.masks = os.listdir(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_path, self.images[index])
        mask_path = os.path.join(self.mask_path, self.masks[index])

        image = np.asarray(Image.open(img_path))
        mask = np.asarray(Image.open(mask_path), dtype=np.float32)
        mask = (mask > 100).astype(float)

        if self.transform is not None:
            output = self.transform(image=image, mask=mask)
            image = output['image']
            mask = output['mask']

        return image, mask


if __name__ == '__main__':
    pass
