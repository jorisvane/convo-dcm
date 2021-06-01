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


dataset = ImageChoiceDataset(csv_file = 'C:/Users/joris/Desktop/BEP PYTHON/dataset.csv', root_dir = '/tudelft.net/staff-umbrella/CNN4DCM/images', transform = my_transforms)

# Test image plot
# plt.imshow(dataset[0][0].permute(1, 2, 0))
# plt.show()

# Data is split in 70, 15, 15

# 45798, 9814, 9813

train_set, test_set, val_set = torch.utils.data.random_split(dataset, [100, 65225, 100])

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
criterion = nn.BCELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Accuracy

# def binary_acc(y_pred, y_test):
#     y_pred_tag = torch.round(y_pred)

#     correct_results_sum = (y_pred_tag == y_test).sum().float()
#     acc = correct_results_sum/y_test.shape[0]
#     acc = torch.round(acc * 100)
    
#     return acc


file = open("Results.txt", "w")

######################

# HYPERPARAMETERS TEXT

######################

file.write('Model 1 : ResNet50 \nbatch size = 10 \nlearning rate = 0.0001 \nnum_epochs = 10\n')

training_acc = []
validation_acc = []

training_loss = []
validation_loss = []

# Train the network
for epoch in range(num_epochs):

    epoch_loss_train = 0.0
    epoch_acc_train = 0.0

    for i, (image1, image2, y_label, price1, price2, delta_cost, delta_rating) in enumerate(train_loader):
        
        # Get data to cuda

        image1 = image1.to(device=device)
        image2 = image2.to(device=device)
        y_label = y_label.to(device=device)
        price1 = price1.to(device=device)
        price2 = price2.to(device=device)
        
        # Forward

        prob = model(image1, price1, image2, price2)
        
        y_label = y_label.unsqueeze(1)
        y_label = y_label.float()

        #print(prob.size())
        #print(y_label.size())

        preds = torch.round(prob)
        total = torch.sum(preds == y_label)

        # print(preds)
        # print(y_label)
        # print(total)

        
        
        loss = criterion(prob, y_label)
        #acc = binary_acc(prob, y_label)

        # Backward
        
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or adam step

        optimizer.step()

        # Calculate loss and accuracy

        epoch_loss_train += loss.item()

        epoch_acc_train += total #acc.item()
    

    # Evaluate model during training
    model.eval()

    with torch.no_grad():

        epoch_loss_val = 0.0
        epoch_acc_val = 0.0

        for j, (image1, image2, y_label, price1, price2, delta_cost, delta_rating) in enumerate(val_loader):
        
        # Get data to cuda

            image1 = image1.to(device=device)
            image2 = image2.to(device=device)
            y_label = y_label.to(device=device)
            price1 = price1.to(device=device)
            price2 = price2.to(device=device)
        
            # Forward

            prob2 = model(image1, price1, image2, price2)

            y_label = y_label.unsqueeze(1)
            y_label = y_label.float()

            preds = torch.round(prob2)
            total = torch.sum(preds == y_label)

            loss = criterion(prob2, y_label)
            #acc = binary_acc(prob2, y_label)

            # Validation loss and accuracy for difference
            
            epoch_loss_val += loss.item()
            epoch_acc_val += total #acc.item()
            


    a = epoch_loss_train / len(train_loader.dataset)
    b = epoch_loss_val / len(val_loader.dataset)
    c = (epoch_acc_train / len(train_loader.dataset)) * 100.
    d = (epoch_acc_val / len(val_loader.dataset)) * 100.
            
    
    file.write(f'Epoch {epoch+1} | Training Loss: {a} | Validation Loss: {b} | Training Accuracy: {c} | Validation Accuracy: {d}\n')
    
    if len(validation_loss) > 0 and b < min(validation_loss):

        torch.save(model.state_dict(), FILE)
        print('Model saved')


    # Appending accuracy and loss for plot

    training_loss.append(a)

    validation_loss.append(b)

    training_acc.append(c)
    
    validation_acc.append(d)


file.close


# Plotting results for each epoch


a_list = list(range(1, num_epochs+1))

fig, (ax1, ax2) = plt.subplots(1, 2)

# Plot 1

ax1.plot(a_list, training_acc, label='Training accuracy')
ax1.plot(a_list, validation_acc, label='Validation accuracy')


# Plot 2

ax2.plot(a_list, training_loss, label='Training loss')
ax2.plot(a_list, validation_loss, label='Validation loss')

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy(%)')
ax1.legend()

ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

fig.tight_layout(pad=3.0)

#################################

# MODEL TYPE AND HYPERPARAMETERS

#################################

fig.suptitle('Model 1 | batchsize : 10 | learning rate : 0.0001')

plt.savefig('Results_plot_training.png')


# Loading and evaluating model