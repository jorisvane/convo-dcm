import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import math

model = models.googlenet(pretrained=True)

print(model)

newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))

pretrained = newmodel

# You could make all the weights from the feature extractor fixed

for parameter in pretrained.parameters():
    parameter.requires_grad = False

# for name, param in pretrained.named_parameters():
#     print(name, param.requires_grad)

# print(pretrained)

class NN(nn.Module):

    def __init__(self, my_pretrained_model):
        super(NN, self).__init__()
        self.pretrained = my_pretrained_model
        
        self.MLP = nn.Sequential(
            nn.Linear(1024, 124),
            nn.ReLU(),                # not in MATLAB model from Sander 
            nn.Linear(124, 1)
        )
        self.last_node = nn.Linear(2,1)
    
    
    def forward_once(self, x, y):

        print(x.size())

        x = self.pretrained(x)

        print(x.size())
        
        x = torch.squeeze(x)
        
        print(x.size())

        x = self.MLP(x)

        print(x.size())

        y = torch.unsqueeze(y,1)
        
        z = torch.cat((x,y), dim=1)
        
        x = self.last_node(z)
        return x
    
    def forward(self, image1, price1, image2, price2):
        output1 = self.forward_once(image1, price1)
        output2 = self.forward_once(image2, price2)
        new_output3 = torch.div(1,torch.add(torch.exp(torch.subtract(output1, output2)),1))

        # output3 = torch.subtract(output1, output2)
        # output4 = torch.exp(output3)
        # output5 = torch.add(output4, 1)
        # output6 = torch.div(1, output5)
        #output6 = 1/(1 + math.exp(output2-output1))    probability of choosing first image
        return new_output3


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model_new = NN(my_pretrained_model=pretrained).to(device)

# print(model_new)

# for name, param in model_new.named_parameters():
#     print(name, param.requires_grad)