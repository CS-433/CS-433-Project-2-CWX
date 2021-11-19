import time
import os
import torch
import torch.optim as optim
import argparse

from model import *
from Losses import *
from make_dataset import *
from utils import *

batch_size_train = 4
batch_size_test = 1

model_name = 'Unet'
save_model = True
model_save_path = 'model/'
tensorboard_path = f'tensorboard/{model_name}/'
Train_Directory = 'training/original/'
Test_Directory = 'test_set_images/'
load_model_name = 'Unet-11_17_14_49.pth'
load_weights_name =  os.path.join('model', load_model_name)

max_epoch = 200
init_lr = 1e-2
# For binarization
thresh1 = 0.2
thresh2 = 0.5
ksize = 11

def train():
    
    description = 'epoch200_imgflip'
    logger = Logger(os.path.join(tensorboard_path, description))


    train_set = MyTrainDataSet(Train_Directory, DataAug=True, isnorm=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size_train, shuffle=True, num_workers=4)
    

    net = Unet(in_ch=3, out_ch=1) 
    net = net.cuda()
    net.train()
    load_last_model=True
    if load_last_model:
        net = torch.load(load_weights_name)


    loss_function = mix_loss(rate=1e5, gamma=2, alpha=.7, eps=1e-7)
    loss_function = loss_function.cuda()
    dice_function = dice_coef()
    dice_function = dice_function.cuda()
    F1_function = F1_coef()
    F1_function = F1_function.cuda()

    optimizer = optim.Adam(net.parameters(), lr=init_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)


    now = time.time()
    for epoch in range(max_epoch):
        print(f'epoch {epoch+1}/{max_epoch} start:')
        print("lr for {}th epoch is:{}".format(epoch+1, optimizer.param_groups[0]['lr']))
        running_loss = 0.
        running_dice_1 = 0.
        running_dice_2 = 0.
        running_F1_1 = 0.
        running_F1_2 = 0.
        
        net.train()
        for i, data in enumerate(train_loader, 0): 
            
            img, mask = data                                   
            img, mask = img.cuda(), mask.cuda()                
            optimizer.zero_grad()                                 
            
            pred = net(img)

            loss = loss_function(pred,mask)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            pred = pred.cpu().detach().numpy()
            pred_1 = torch.from_numpy(binarize(pred, thresh1)).cuda()
            pred_2 = torch.from_numpy(binarize(pred, thresh2)).cuda()

            dice_1 = dice_function(pred_1,mask)
            dice_2 = dice_function(pred_2,mask)
            running_dice_1 += dice_1.item()
            running_dice_2 += dice_2.item()
            F1score_1 = F1_function(pred_1,mask)
            F1score_2 = F1_function(pred_2,mask)
            running_F1_1 += F1score_1.item()
            running_F1_2 += F1score_2.item()

            consume_time = time.time() - now
            now = time.time()

            print("Step {}/{},  loss:{:.2f}, dice_1:{:.3f}, dice_2:{:.3f}, F1score_1:{:.3f}, F1score_2:{:.3f}, spent time:{:.2f}s"
            .format(i+1, len(train_loader), loss.item(), dice_1.item(), dice_2.item(), F1score_1.item(), F1score_2.item(), consume_time),'\n')

        scheduler.step()
  
        scalarinfo = {
            'loss':running_loss/(i+1),
            'dice_1': running_dice_1/(i+1),
            'dice_2': running_dice_2/(i+1),
            'f1score_1': running_F1_1/(i+1),
            'f1score_2': running_F1_2/(i+1),
        }

        for tag, value in scalarinfo.items():
            if isinstance(value,(list, tuple)):
                for va in value:                                                      
                    logger.scalar_summary(tag, va, epoch+1)
            else:
                logger.scalar_summary(tag, value, epoch+1)


    if save_model:
        now = time.time()
        timeArray = time.localtime(now)
        timeStr = time.strftime("%m_%d_%H_%M", timeArray)
        model_name_time = "{}-{}".format(model_name, timeStr)
        weights_name = os.path.join(model_save_path, model_name_time+'-'+description+'.pth')

        torch.save(net, weights_name)


def test():

    test_set = MyTestDataSet(Test_Directory, isnorm=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size_test, shuffle=False, num_workers=4)
    

    load_last_model = True
    if load_last_model:
        net = torch.load(load_weights_name)

    net.eval()
    for i, data in enumerate(test_loader, 0):
        print('testdata_{} start to be processed'.format(i+1))
        img = data
        img = img.cuda()

        with torch.no_grad():
            pred = net(img)
            pred = pred.cpu().numpy()
            img = img.cpu().numpy()

            # binarization
            pred_1 = binarize(pred, thresh1)
            pred_2 = binarize(pred, thresh2)  

            for j in range(pred.shape[0]):
                image = img[j]
                pred_img1 = pred_1[j] * 255. 
                pred_img2 = pred_2[j] * 255.

                image = image.swapaxes(0, 2)
                pred_img1 = pred_img1.swapaxes(0, 2)  
                pred_img2 = pred_img2.swapaxes(0, 2)

                pred_img1 = post_process(pred_img1, ksize)
                pred_img2 = post_process(pred_img2, ksize)

                index = batch_size_test*i+j+1
                save_img(pred_img1, load_model_name, thresh1, index)
                save_img(pred_img2, load_model_name, thresh2, index)
                save_img(image, load_model_name, thresh1, index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--ParA", help="mode 1 for train, mode 2 for test",type=int)
    args = parser.parse_args()
    mode = args.ParA

    if mode == 1:
        train()
    elif mode == 2:
        test()
    else:
        raise ValueError('Not correct mode num')