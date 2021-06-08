import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageChoiceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=True): # transform anders?
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path1 = os.path.join(self.root_dir, self.annotations.iloc[index, 1]) # name first image in first column
        img_path2 = os.path.join(self.root_dir, self.annotations.iloc[index, 2]) # name second images in second column
        
        image1 = io.imread(img_path1)
        print('image 1 loaded')
        image2 = io.imread(img_path2)
        print('image 2 loaded')
        
        # image1 = Image.open(img_path1)
        # image2 = Image.open(img_path2)

        y_label = torch.tensor(int(self.annotations.iloc[index, 3])) # choice made in third column
        
        price1 = torch.tensor(float(self.annotations.iloc[index, 4])) # price first image
        price2 = torch.tensor(float(self.annotations.iloc[index, 5])) # price second image

        # DIT TOEVOEGEN
        
        delta_cost = torch.tensor(float(self.annotations.iloc[index, 6])) # delta cost
        delta_rating = torch.tensor(float(self.annotations.iloc[index, 7])) # delta price

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

            if image1.shape[0]==1:
                image1 = image1.repeat(3,1,1)

            if image2.shape[0]==1:
                image2 = image2.repeat(3,1,1)


        return [image1, image2, y_label, price1, price2, delta_cost, delta_rating]
