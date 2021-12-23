# CS-433 Machine Learning Project 2

### A LinkNet Based Model to Classify Statellite Images Segments

#  

__Team Member (CWX)__: 
  * [Zewei Xu](mailto:zewei.xu@epfl.ch)
  * [Haoxuan Wang](mailto:haoxuan.wang@epfl.ch)
  * [Ganyuan Cao](mailto:ganyuan.cao@epfl.ch)

## Project Description
We implemented a model based on the LinkNet architecture to classify the road segments with the satellite images from Google Maps. We evaluate the performance of LinkNet, U-Net and their variants. We finally decided to develop our model on the top of LinkNet architecture. 

In addition, we implemented U-Net architecture (and its variants: U-Net++) as a comparison to show the performance of our model. 


## File/Folder Description
`/data`: The folder that contains the training and the testing images, it is also used to save the prediction masks.

`/log`: The folder that contains the evaluation results of different models.

`Losses.py`: Loss Functions including BCE Loss, Dice Loss, Focal Loss and a mix loss combined by Focal Loss and Dice Loss.

`dataset.py`: Load images and perform data augment on images.

`Helper.py`: Some helper functions and image transformation.

`mask_to_submission.py`: Convert the prediction results into a submission file.

`model.py`: Model implementation.
* Unet
* LinkNet
* LinkNet1
* DinkNet
* UnetPP
* LinkNetPP
* DoubleUnet

`run.py`: It creates a model and train it, or loads a pre-trained model, and it makes predictions on the test set.

`submission_to_mask.py`: Convert the submission file into prediction images.

`train.py`: It trains the model.

`utile.py`: It contains some utile functions, such as:
* get_loader: Create the DataLoader class
* f1: Compute F1 score
* accuracy: Compute F1 score, Precision and Recall
* create_different_prospective: Apply transformations to the image and return different prospectives
* combine_different_prospective: Combine predictions of different prospectives
* img_crop: Crop the image into patches of a given size
* combine_img: Combine cropped patches into the original image

`visualization.ipynb`: A visualization of performance of different models. 

## Running requirements
### Environment
* Python 3.8.5
* Jupyter Notebook 6.1.4
### Required libraries
* Numpy 1.20.3
* PyTorch 1.8.2
* PIL 8.2.0
* albumentations 1.1.0
* tqdm 4.59.0
* matplotlib 3.3.4
* torchsummary
* pandas 1.2.4
* torchvision

To install all the required libraries, you can run the following command from the terminal.
```sh
pip install -r requirement
```

## Checkpoints
Our pre-trained models can be found here:

https://drive.google.com/drive/folders/1VPxJjSSlY1VeJWWgAL2eXIICsWvUjpjo?usp=sharing
