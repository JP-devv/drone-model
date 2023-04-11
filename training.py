import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import cv2
import albumentations as A

import time
import os
from tqdm.notebook import tqdm

from torchsummary import summary
import segmentation_models_pytorch as smp

device = torch.device("cpu")

IMAGE_PATH = '/Users/johannplaster/Desktop/ \
github/drone-model/dataset-package/dataset/ \
semantic_drone_dataset/original_images/'

MASK_PATH = '/Users/johannplaster/Desktop/ \
github/drone-model/dataset-package/dataset/ \
semantic_drone_dataset/label_images_semantic/'
