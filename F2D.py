import sys
sys.path.append('./utils')
sys.path.append('./src/Models')
import upgraded_net_hook
import dataloader
import torch
import numpy as np


"""
from torchvision import models
from torchsummary import summary

model = upgraded_net_hook.ResNet(upgraded_net_hook.ResidualBlock, [3, 4, 6, 3])"""


def test():
    net = upgraded_net_hook.ResNet20()
    y = net(torch.randn(1, 1, 28, 28))

test()

"""

def few_two_decide (ResNet_Model, dataloader):
    model = ResNet_Model()
    #y = net(torch.randn(1, 1, 28, 28))
    #rint(y.size())
        

    #dataloader = torch.rand(1,1,28,28)

    #model = upgraded_net_hook.simple_net_upgraded(1, 10) 
    model.linear.register_forward_hook(upgraded_net_hook.get_activation('linear'))
    model.avg_pooling.register_forward_hook(upgraded_net_hook.get_pooling('avg_pooling'))
    last_convolution = model(dataloader)                      #get weight matrix

    #print(upgraded_net_hook.activation['fc2'])
    #print(upgraded_net_hook.average_pooling['avg_pool2d'])#get average pooling
    weight_matrix = upgraded_net_hook.activation['linear']
    avg_matrix = upgraded_net_hook.average_pooling['avg_pooling']
    print(weight_matrix.shape)
    print(avg_matrix.shape)
    #print(torch.transpose(weight_matrix))
    print("---Hadamard product---")
    values_mul = torch.mul(weight_matrix,avg_matrix) # 1. Hadamard Multiplication Elementwise Multiplication of matrices (the shape should remain the same)
    #print(values_mul)
    print("--sorted---")
    values_sort, index = torch.sort(values_mul, dim=1) # 2. sort the connections calculation results of each neuron from min to max and get the V2
    print(values_sort)

    print("---Clipping---")

    min_quantile = torch.quantile(values_sort, 0.3).data.tolist()
    max_quantile = torch.quantile(values_sort, 0.6).data.tolist()

    values_clip = torch.clamp(values_sort,min=min_quantile, max=max_quantile, out=None) # In this step the nd Tensor should get the top and bottom 30 percent of the values set to zero

    print(values_clip)

    print("---Sum---")


    values_sum = torch.sum(values_clip,1) #Prediction Score

    print(values_sum)


few_two_decide(upgraded_net_hook.ResNet56, torch.rand(1,1,28,28) )"""