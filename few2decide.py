
import sys
sys.path.append('./utils')
sys.path.append('./src/Models')
import upgraded_net_hook
import torch
import numpy as np

from torchvision import models
from torchsummary import summary


#summary(upgraded_net_hook.simple_net_upgraded(1, 10) ,(1,28,28))

x = torch.rand(1,1,28,28)
model = upgraded_net_hook.simple_net_upgraded(1, 10) 
model.fc2.register_forward_hook(upgraded_net_hook.get_activation('fc2'))
model.avg_pool2d.register_forward_hook(upgraded_net_hook.get_pooling('avg_pool2d'))
last_convolution = model(x)                      #get weight matrix

#print(upgraded_net_hook.activation['fc2'])
#print(upgraded_net_hook.average_pooling['avg_pool2d'])#get average pooling
weight_matrix = upgraded_net_hook.activation['fc2']
avg_matrix = upgraded_net_hook.average_pooling['avg_pool2d']
print(weight_matrix.shape)
print(avg_matrix.shape)
#print(torch.transpose(weight_matrix))

#v1 = weight_matrix*avg_matrix # 1. Hadamard Multiplication Elementwise Multiplication of matrices (the shape should remain the same)


#v2 = torch.sort(v1, dim=1, descending=False, stable=False, out=None) # 2. sort the connections calculation results of each neuron from min to max and get the V2


#v3 = torch.clamp(v1,min=-0.5, max=0.5) # In this step the nd Tensor should get the top and bottom 30 percent of the values set to zero


#V4 = torch.sum(v1,1) #Prediction Score

