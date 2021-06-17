import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from CustomDataset import ImageChoiceDataset
from evaluation import function_eval

from model import NN
from model import pretrained

import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

import itertools
import numpy as np
from sklearn.metrics import log_loss

print('START')

# Loading and evaluating model

cwd = os.getcwd()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 10
learning_rate = 0.0001
num_epochs = 10

#Load and transform Data

my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    
    #transforms.Normalize(mean=[0.0,0.0,0.0], std=[1.0,1.0,1.0]) # This does nothing

])


#########################################################

# PATH TO CSV AND PATH TO IMAGES FOLDER IMPORTANT FOR HPC

#########################################################


dataset = ImageChoiceDataset(csv_file = 'dataset.csv', root_dir = '/tudelft.net/staff-umbrella/CNN4DCM/images4AVA/images', transform = my_transforms)

# Test image plot
# plt.imshow(dataset[0][0].permute(1, 2, 0))
# plt.show()

# Data is split in 70, 15, 15

# 45798, 9814, 9813

train_set, test_set, val_set = torch.utils.data.random_split(dataset, [100, 100, 65225])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

model = NN(my_pretrained_model=pretrained).to(device)

FILE = cwd + '/model1st.pth'

model.load_state_dict(torch.load(FILE))
model.eval()

delta_cost_eval = []

delta_rating_eval = []

prob_eval = []

y_label_eval = []

for i, (image1, image2, y_label, price1, price2, delta_cost, delta_rating) in enumerate(test_loader):

    
    image1 = image1.to(device=device)
    image2 = image2.to(device=device)
    y_label = y_label.to(device=device)
    price1 = price1.to(device=device)
    price2 = price2.to(device=device)
    delta_cost = delta_cost.to(device=device)
    delta_rating = delta_rating.to(device=device)

    # Forward

    prob = model(image1, price1, image2, price2)

    a = delta_cost.tolist()
    b = delta_rating.tolist()

    d = y_label.tolist()
    
    c = prob.tolist()
    List_flat = list(itertools.chain(*c))
    

    delta_cost_eval.append(a)
    
    delta_rating_eval.append(b)
    
    prob_eval.append(List_flat)

    y_label_eval.append(d)



y_label_eval = list(itertools.chain(*y_label_eval))

delta_cost_eval = list(itertools.chain(*delta_cost_eval))

delta_rating_eval = list(itertools.chain(*delta_rating_eval))

prob_eval = list(itertools.chain(*prob_eval))


y_label_eval = np.array(y_label_eval)

delta_cost_eval = np.array(delta_cost_eval)

delta_rating_eval = np.array(delta_rating_eval)

prob_eval = np.array(prob_eval)

name = 'Model 1 : ResNet50 : ratio'

ratio, params = function_eval(delta_cost_eval, delta_rating_eval, prob_eval, name)

LL = -log_loss(y_label_eval, prob_eval, normalize=False)

cross_entropy = -LL/len(y_label_eval)

rho_square = 1-(LL/(len(y_label_eval)* np.log(0.5)))

file = open("Results_eval_ResNet50_TEST.txt", "w")

print(f'Parameters: {params}')
print(f'Ratio: {ratio}')
print(f'Log loss: {LL}')
print(f'Cross entropy: {cross_entropy}')
print(f'Rho: {rho_square}')

file.write(f'Model 1 TEST : ResNet50 \nbatch size = 10 \nlearning rate = 0.0001 \nnum_epochs = 10\n Parameters: {params} \nRatio: {ratio} \nLog loss: {LL} \nCross entropy: {cross_entropy} \nRho: {rho_square}')

file.close()

print('DONE')