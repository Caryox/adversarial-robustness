import sys
sys.path.append('./utils')
sys.path.append('./src/Models')
import upgraded_net_hook
import dataloader
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import device
from torch.autograd import Variable

"""
from torchvision import models
from torchsummary import summary

model = upgraded_net_hook.ResNet(upgraded_net_hook.ResidualBlock, [3, 4, 6, 3])"""

def few_two_decide(model, dataloader):
    model.eval()
    
    model.linear.register_forward_hook(upgraded_net_hook.get_activation('linear'))
    model.avg_pooling.register_forward_hook(upgraded_net_hook.get_pooling('avg_pooling'))
    sample = next(iter(dataloader))
    inputs, labels = sample
    inputs, labels = Variable(inputs.to(device.device)), Variable(labels.to(device.device))
    pred = model(inputs)

    print("Get matrix")
    weight_matrix = upgraded_net_hook.activation['linear']
    avg_matrix = upgraded_net_hook.average_pooling['avg_pooling']
    print("---Hadamard product---")
    values_mul = torch.mul(weight_matrix,avg_matrix[0]) # 1. Hadamard Multiplication Elementwise Multiplication of matrices (the shape should remain the same)
    print("--sorted---")
    values_sort, index = torch.sort(values_mul, dim=0) # 2. sort the connections calculation results of each neuron from min to max and get the V2

    print("---Clipping---")

    min_quantile = torch.quantile(values_sort, 0.3).data.tolist()
    max_quantile = torch.quantile(values_sort, 0.6).data.tolist()

    values_clip = torch.clamp(values_sort,min=min_quantile, max=max_quantile, out=None) # In this step the nd Tensor should get the top and bottom 30 percent of the values set to zero

    print("---Sum---")

    #values_sum = torch.sum(values_clip, dim=0) #Prediction Score
    values_sum = values_clip.sum(0)
    
    return values_sum, labels



def test(skip = False):
    num_epochs=2
    lr = 0.0002
    momentum = 0.9
    w_decay = 0.001
    milestones= [50,75]
    gamma = 0.1
    model = upgraded_net_hook.ResNet20().to(device.device)
    trainloader = dataloader.train_dataloader
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
    loss_func = nn.CrossEntropyLoss().to(device.device)
    if skip:
        model_point = torch.load("./utils/few2decide_model.tar", map_location=device.device)
        model.load_state_dict(model_point["state_dict"])
    else:
        model.train()
        for epoch in range(num_epochs):
            print("Epoch: ", epoch)
            for i, data in enumerate(trainloader):
                inputs, labels = data
                inputs, labels = Variable(inputs.to(device.device)), Variable(labels.to(device.device))
                pred = model(inputs)
                loss = loss_func(pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred = pred.data.max(1, keepdim=True)[1]
        torch.save({"state_dict": model.state_dict()}, "./utils/few2decide_model.tar")
    return few_two_decide(model, trainloader)

pred, labels = test(True)

#print(sums)
print(pred.shape)
print(labels.shape)
for i in range(len(labels)):
    print(pred[i].argmax(), labels[i])
    print("\n")

# To Do
"""from sklearn.metrics import accuracy_score
print(accuracy_score(labels.item(), pred.argmax().item()))
#print("Few2Decide :" + str(sums[0].argmax()) + " " + str(pred[0]))
"""