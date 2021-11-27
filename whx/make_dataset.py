import os
import numpy as np
import cv2
import torch
from torch.utils import data
from natsort import natsorted
from utils import binarize

class MyTrainDataSet(data.Dataset):
    
    def __init__(self, root, DataAug=True, isnorm=False):
        super(MyTrainDataSet,self).__init__()
        
        self.root = root              
        self.DataAug=DataAug
        self.isnorm=isnorm
        self.img_names = [item for item in os.listdir(self.root + 'images/')]
    
    def rgb2gray(self,rgb_img):
        # Y=0.2989∗R + 0.5870∗G + 0.1140∗B
        gray_img = 0.2989*rgb_img[0] + 0.5870*rgb_img[1] + 0.1141*rgb_img[2]
        return gray_img

    def get_image(self,name):
        try:
            img_path = os.path.join(self.root, 'images', name)
            label_path = os.path.join(self.root, 'groundtruth', name)
            img = cv2.imread(img_path)
            label = cv2.imread(label_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)


            if self.DataAug:
                axs=np.random.randint(0,2,size=3)
                for i in range(0,3):
                    if axs[i]: 
                        img = cv2.flip(img, i-1)
                        label = cv2.flip(label, i-1)
                        break
                    else:
                        continue

            # opencv 彩色图像维度是(w, h, 3)
            img = img.swapaxes(0, 2)
            if self.isnorm:
                img = img.astype(np.float)
                for i in range(3):
                    img[i] -= np.mean(img[i])
                    img[i] /= np.std(img[i])
                
            label = label.swapaxes(0, 2)
            label = self.rgb2gray(label)

        except FileNotFoundError:
            print('Unable to find {} file'.format(name))
            return None 
        
        return img, label

    def __getitem__(self, index):
        name = self.img_names[index]
        img, label = self.get_image(name)
        print(np.max(img))

        label = label[np.newaxis,:,:]
        label = binarize(label, 0.1)


        return torch.from_numpy(img).type(torch.float32), torch.from_numpy(label).type(torch.float32)
    
    def __len__(self):
        return len(self.img_names)

class MyTestDataSet(data.Dataset):

    def __init__(self, root, isnorm=False):
        super(MyTestDataSet, self).__init__()

        self.root = root
        self.isnorm = isnorm
        self.img_names = natsorted([item for item in os.listdir(self.root)])

    def get_image(self,name):
        try:
            img_path = os.path.join(self.root, name, name+'.png')
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.swapaxes(0, 2)

            if self.isnorm:
                img = img.astype(np.float)
                for i in range(3):
                    img[i] -= np.mean(img[i])
                    img[i] /= np.std(img[i])


        except FileNotFoundError:
            print('Unable to find {} file'.format(name))
            return None 
        
        return img

    def __getitem__(self, index):
        name = self.img_names[index]
        img = self.get_image(name)

        return torch.from_numpy(img).type(torch.float32)
    
    def __len__(self):
        return len(self.img_names)


if __name__ == '__main__':

    TrainData_Directory = 'D:/EPFL/GitHub/road-segmentation/training/original/'
    TestData_Directory = 'D:/EPFL/GitHub/road-segmentation/test_set_images/'
    plan = 1

    if plan == 1:
        dst = MyTrainDataSet(TrainData_Directory, DataAug=True)
        Batch_size = 1
        train_loader = data.DataLoader(dst, batch_size = Batch_size)

        for i, data in enumerate(train_loader, 0):
            print(i+1,' start')
            imgs, labels = data

            print('imgs.max={},imgs.min={},imgs.mean={}'.format(np.max(imgs.numpy()),np.min(imgs.numpy()),np.mean(imgs.numpy())))
            print('labels.max={},labels.min={}'.format(np.max(labels.numpy()),np.min(labels.numpy())))
            print('imgs.shape={},labels.shape={}'.format(imgs.shape,labels.shape),'\n')

    if plan == 2:
        dst =MyTestDataSet(TestData_Directory)
        Batch_size = 1
        test_loader = data.DataLoader(dst, batch_size = Batch_size)

        for i, data in enumerate(test_loader, 0):
            print(i+1,' start')
            imgs = data
            # print('imgs.max={},imgs.min={}'.format(np.max(imgs.numpy()),np.min(imgs.numpy())))
            # print('imgs.shape={}'.format(imgs.shape),'\n')
                   