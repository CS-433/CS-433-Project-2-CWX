import numpy as np
import torch.nn as nn
import torch
import os
import cv2
from tensorboardX import SummaryWriter

class Logger(object):
    
    def __init__(self, log_dir):
        '''Create a summary writer logging to log_dir.'''
        self.writer = SummaryWriter(log_dir)


    def scalar_summary(self, tag, value, step):
        '''Log a scalar variable.'''
        self.writer.add_scalar(tag, value, global_step=step)

class F1_coef(nn.Module):
    '''
    Compute F1 Score
    '''
    def __init__(self):
        super(F1_coef, self).__init__()
    def forward(self, pred, label):
        pred = pred.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        #true positive
        TP = np.sum(np.logical_and(np.equal(label,1),np.equal(pred,1)))
 
        #false positive
        FP = np.sum(np.logical_and(np.equal(label,0),np.equal(pred,1)))
 
        #false negative
        FN = np.sum(np.logical_and(np.equal(label,1),np.equal(pred,0)))
 
        #true negative
        TN = np.sum(np.logical_and(np.equal(label,0),np.equal(pred,0)))

        precision = TP / (TP + FP + 1e-5)
        recall = TP / (TP + FN + 1e-5)
        F1score = 2*recall*precision / (recall + precision + 1e-5)

        return torch.tensor(F1score).cuda()

def binarize(a,thresh):
    '''
    Set a threshold to do the binarization
    '''
    a = a.astype(np.float)
    a -= np.min(a)
    a /= np.max(a)
    
    b = np.where(a<thresh, 0, 1)

    return b

def make_dirs(path_list):
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)

def resize_img(path):
    for item in os.listdir(os.path.join(path, 'original', 'images')):
        img_path = os.path.join(path, 'original', 'images', item)
        label_path = os.path.join(path, 'original', 'groundtruth', item)

        img = cv2.imread(img_path)
        label = cv2.imread(label_path)

        img = cv2.resize(img, dsize=(608,608), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, dsize=(608,608), interpolation=cv2.INTER_LINEAR)

        img_save_path = img_path.replace('original', 'resize_training')
        label_save_path = label_path.replace('original', 'resize_training')

        make_dirs([os.path.dirname(img_save_path), os.path.dirname(label_save_path)])

        cv2.imwrite(img_save_path, img)
        cv2.imwrite(label_save_path, label)

def post_process(img, ksize):
    kernel1 = np.ones((ksize, ksize), np.uint8)
    kernel2 = np.ones(((ksize-1)//2, (ksize-1)//2), np.uint8)
    img1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)
    img2 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel2)

    return img2

def save_img(img, model_name, thresh, index):
    if img.ndim==2:
        save_path = os.path.join('result', model_name.split('.')[0], f'label_thresh_{thresh}')
        make_dirs([save_path])
        cv2.imwrite('{}/test_{}.png'.format(save_path,index), img)

    elif img.ndim==3 and img.shape[2]==3:
        save_path = os.path.join('result', model_name.split('.')[0], 'image')
        make_dirs([save_path])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('{}/test_{}.png'.format(save_path,index), img)

    else:
        raise ValueError('Invalid img type')


if __name__ == '__main__':

    resize_img('D:/EPFL/GitHub/road-segmentation/training/')