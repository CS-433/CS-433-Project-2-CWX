# CS-433-Project-2-CWX
__Team Member__: 
  * [Zewei Xu](mailto:zewei.xu@epfl.ch)
  * [Haoxuan Wang](mailto:haoxuan.wang@epfl.ch)
  * [Ganyuan Cao](mailto:ganyuan.cao@epfl.ch)

## Project Description
We implemented a model based on the LinkNet architecture to classify the road segments with the satellite images from Google Map. We evaluate the performance of LinkNet, U-Net and their variants. We finally decided to develop our model on the top of LinkNet architecture. 

In addition, we implemented U-Net architecture (and its variants: U-Net++) as a comparision to show the performance of our model. 


## File/Folder Description
`/data`: The folder that contains the training and the testing images.
`/log`: The folder that contains the evaluation results of different models.
`/whx`: The U-Net model implemented by Haoxuan Wang.
`Losses.py`: Loss Functions including BCE Loss, Dice Loss, Focal Loss and a mix loss combined by Focal Loss and Dice Loss.
`dataset.py`: Load images and perform data augment on images.
'Helper.py`: Some helper functions and image transformation
`mask_to_submission.py`:
`model.py`: model implementation
`run.py`:
`submission_to_mask.py`:
`train.py`:
`utile.py`:
`visualization.ipynb`: A visualization of performance of different models. 

## Requirement
* Python 3.8.5
* Jupyter Notebook 6.1.4
* Numpy 1.20.3
* PyTorch 1.8.2


## Checkpoints
https://drive.google.com/drive/folders/1VPxJjSSlY1VeJWWgAL2eXIICsWvUjpjo?usp=sharing
