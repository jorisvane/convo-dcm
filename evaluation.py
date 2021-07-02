# %%
from mpl_toolkits import mplot3d
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# Import functions from ResNet, GoogleNet, DenseNet and Inceptionv3

import torch
from GoogleNet import pretrained
from GoogleNet import NN
import os


# x_data = [8,4,3,5]
# y_data = [2,8,2,6] 
# z_data = [4,6,7,10]


def function_eval(x_data,y_data,z_data,name):

    # x_data = np.array(x_data)
    # y_data = np.array(y_data)
    # z_data = np.array(z_data)


    # Fit a function through the data to represtent the ratio


    def func(X, a, b):
        x,y = X[0], X[1]

        return 1./(1 + np.exp(x*a+y*b))

    params, covar = curve_fit(func, (x_data,y_data), z_data)

    ratio = -params[1]/params[0]


    z_data_func = func((x_data, y_data), params[0], params[1])


    # Creating radii and angles
    r = np.linspace(0.125, 1.0, 100)  
    a = np.linspace(0, 2 * np.pi, 
                    100,
                    endpoint = False)  
        
    # Repeating all angles for every radius  
    a = np.repeat(a[..., np.newaxis], 100, axis = 1)
        
    # Creating figure
    fig = plt.figure(figsize =(16, 9))  
    ax = plt.axes(projection ='3d') 

    ax.scatter3D(x_data, y_data, z_data)
    
    # Creating color map
    my_cmap = plt.get_cmap('hot')
        
    # Creating plot
    trisurf = ax.plot_trisurf(x_data, y_data, z_data_func,
                            cmap = my_cmap,
                            linewidth = 0.2, 
                            antialiased = True,
                            edgecolor = 'grey')  
    fig.colorbar(trisurf, ax = ax, shrink = 0.5, aspect = 5)
    ax.set_title(name)
    
    # Adding labels
    ax.set_xlabel('delta cost', fontweight ='bold') 
    ax.set_ylabel('delta rating', fontweight ='bold') 
    ax.set_zlabel('probability', fontweight ='bold')
        
    plt.savefig('Evaluation_DenseNet_last_layer.png', dpi=300)

    # show plot
    plt.show()

    return ratio, params

# function_eval(x_data,y_data,z_data)
# %%
