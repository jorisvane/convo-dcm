import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

from CustomDataset import ImageChoiceDataset

from model import NN
from model import pretrained

import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

cwd = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 10
learning_rate = 0.0001
num_epochs = 10

my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    
    #transforms.Normalize(mean=[0.0,0.0,0.0], std=[1.0,1.0,1.0]) # This does nothing

])

dataset = ImageChoiceDataset(csv_file = 'dataset_test.csv', root_dir = '/tudelft.net/staff-umbrella/CNN4DCM/images', transform = my_transforms)

train_set, test_set, val_set = torch.utils.data.random_split(dataset, [8, 0, 0])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

print(train_loader)