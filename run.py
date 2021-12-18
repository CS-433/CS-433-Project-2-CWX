import os
import albumentations
from PIL import Image
from albumentations import *
from tqdm import tqdm

import train
from mask_to_submission import masks_to_submission
from utile import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Image_size = 608
# Test Time Augmentation: predict the image by applying different augmentation operations on it,
# and determine the final output by averaging the results
rotation = [0, 90, 180, 270]
transpose = True
one_hot = False
pretrained = True
# Load the pre-trained model
model_name = 'final.pth'

if __name__ == '__main__':
    if pretrained:
        model = torch.load('Checkpoints/' + model_name).to(device)
        model.eval()
    else:
        model = train.main('model')
        model.eval()
    # Create the directory to store the predictions
    path = 'data/test_set_images'
    pred_path = 'data/prediction/' + model_name
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    # For each image, apply Test Time Augmentation, make predictions and save predictions
    images = os.listdir(path)
    for image in tqdm(images):
        img_path = os.path.join(path, image, image + '.png')
        im = np.asarray(Image.open(img_path)) / 255
        ims = create_different_prospective([im], rotation, transpose)

        with torch.no_grad():
            output = model(ims.to(device))
            output = output[:, 0].unsqueeze(1)
            predicts = torch.sigmoid(output).cpu().detach()
            # predicts = torch.sigmoid(model(ims.to(device))).cpu().detach()

        if one_hot:
            predicts = predicts.argmax(1).unsqueeze(1).float()
        predict = combine_different_prospective(predicts.numpy(), rotation, transpose). \
            reshape((608, 608))
        # predict[predict < 0.5] = 0
        # predict[predict >= 0.5] = 1
        predict *= 255
        Image.fromarray(predict).convert('L').save(os.path.join(pred_path, image) + '.png')

    # Generate the submission file
    submission_filename = 'dummy_submission.csv'
    image_filenames = []
    for i in range(1, 51):
        image_filename = pred_path + '/test_' + str(i) + '.png'
        print(image_filename)
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)
