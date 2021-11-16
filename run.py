import os
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import numpy as np
from utile import *

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

model = torch.load('Unet_max_f1.pth').to('cuda')

transform = alb.Compose(
    [
        alb.Resize(height=Image_size, width=Image_size),
        alb.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ToTensorV2()
    ]
)

path = 'data/test_set_images'
images = os.listdir(path)
for image in images:
    img_path = os.path.join(path, image, image + '.png')
    im = np.asarray(Image.open(img_path))
    im = transform(image=im)['image'].to('cuda')
    with torch.no_grad():
        pred = torch.sigmoid(model(im.reshape((1, 3, 608, 608))))
    pred = pred.cpu().numpy()
    pred = 255.0*pred
    pred_im = Image.fromarray(pred.reshape((608, 608))).convert('L')
    pred_im.save('prediction/' + image + '.png')

