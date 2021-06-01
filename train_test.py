#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
# import torchvision
from torch.utils.data import DataLoader

from CustomDataset import ImageChoiceDataset

from model import NN
from model import pretrained

import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os


import time
import copy


cwd = os.getcwd()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 10
learning_rate = 0.0001
num_epochs = 2

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


dataset = ImageChoiceDataset(csv_file = 'C:/Users/joris/Desktop/BEP PYTHON/dataset.csv', root_dir = 'C:/Users/joris/Desktop/images4AVA/images4AVA/images', transform = my_transforms)

# Test image plot
# plt.imshow(dataset[0][0].permute(1, 2, 0))
# plt.show()

# Data is split in 70, 15, 15

train_set, test_set, val_set = torch.utils.data.random_split(dataset, [45798, 9814, 9813])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


model = NN(my_pretrained_model=pretrained).to(device)

##########################

# NAME MODEL VERSION

##########################

FILE = cwd + '/model.pth'

torch.save(model.state_dict(), FILE)

# CUDA ran out of memory so set allot smaller batch size and tried emptying the cache
#torch.cuda.empty_cache()

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Accuracy

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


file = open("Results.txt", "w")

######################

# HYPERPARAMETERS TEXT

######################

file.write('Model 1 : ResNet50 \nbatch size = 10 \nlearning rate = 0.0001 \nnum_epochs = 10\n')

training_acc = []
validation_acc = []

training_loss = []
validation_loss = []


best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 0.0

#%%
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    # train

    model.train()
    running_loss_train = 0.0
    running_corrects_train = 0.0

    for i, (image1, image2, y_label, price1, price2, delta_cost, delta_rating) in enumerate(train_loader):
        image1 = image1.to(device=device)
        image2 = image2.to(device=device)
        y_label = y_label.to(device=device)
        price1 = price1.to(device=device)
        price2 = price2.to(device=device)

        prob = model(image1, price1, image2, price2)

        preds = torch.round(prob)
        
        print(preds)

        y_label = y_label.unsqueeze(1)
        y_label = y_label.float()
        
        print(y_label)

        loss = criterion(prob, y_label)
        loss.backward()
        optimizer.step()

        running_loss_train += loss.item() * y_label.size(0)

        print(torch.sum(preds == y_label.data))

        running_corrects_train += torch.sum(preds == y_label.data)

    epoch_loss_train = running_loss_train / len(train_loader)
    epoch_acc_train = running_corrects_train / len(train_loader)


    with torch.no_grad():

        model.eval()
        running_loss_eval = 0.0
        running_corrects_eval = 0.0
        for i, (image1, image2, y_label, price1, price2, delta_cost, delta_rating) in enumerate(val_loader):
            image1 = image1.to(device=device)
            image2 = image2.to(device=device)
            y_label = y_label.to(device=device)
            price1 = price1.to(device=device)
            price2 = price2.to(device=device)

            prob = model(image1, price1, image2, price2)

            preds = torch.round(prob)

            

            y_label = y_label.unsqueeze(1)
            y_label = y_label.float()

            

        
            loss = criterion(prob, y_label)
            
            
            running_loss_eval += loss.item() * y_label.size(0)
            
            
            
            running_corrects_eval += torch.sum(preds == y_label.data)
    
    epoch_loss_val = running_loss_eval / len(val_loader)
    epoch_acc_val = running_corrects_eval / len(val_loader)

    if epoch_loss_val < best_loss:
        epoch_loss_val = best_loss
        best_model_wts = copy.deepcopy(model.state_dict())

    # Test number corrects
    
    print('Loss train: {:.4f} Acc train: {:.4f} Loss val: {:.4f} Acc val: {:.4f}'.format(
                epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val))


# %%
