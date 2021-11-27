import os

import torch
from tqdm import tqdm
from utile import *
from whx.utils import post_process

Train_image_path = 'data/training/images'
Train_mask_path = 'data/training/groundtruth'
Test_image_path = 'data/testing/images'
Test_mask_path = 'data/testing/groundtruth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Learning_rate = 1e-4
Batch_size = 4  # 64
Num_epochs = 500
Num_workers = 0
Image_size = 608

Pin_memory = True
path = 'Checkpoints/'
model_name = 'LinkNet_resnet152_max_f1'

model = torch.load(path + model_name + '.pth').to('cuda')

path = 'data/test_set_images'
images = os.listdir(path)
for image in tqdm(images):
    img_path = os.path.join(path, image, image + '.png')
    im = np.asarray(Image.open(img_path))
    ims = img_crop([im], Image_size, stride=120, padding=0)
    preds = []
    for i in range(len(ims)):
        im = np.array(ims[i], dtype=np.float32)
        with torch.no_grad():
            pred_im = different_prospective_prediction(im, model, rotations=[0,90,180,270], transposes=True)
        preds.append(pred_im)

    preds = np.asarray(preds)
    pred_im = combine_img(preds, 608, stride=120, padding=0)[0]
    pred_im = Image.fromarray(pred_im).convert('L')

    pred_path = 'data/prediction/' + model_name
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    pred_im.save(os.path.join(pred_path, image) + '.png')
