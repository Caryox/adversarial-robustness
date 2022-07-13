import sys
from typing import final

#import torchmetrics
sys.path.append('./utils')
sys.path.append('././src/Models')
import upgraded_net_hook
import param
import dataloader
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import device
from torch.autograd import Variable


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
    #avg_matrix = avg_matrix.mean(dim=(-2, -1))
    
    #print(weight_matrix.size())

    #print(avg_matrix.size())
    
    print("---Hadamard product---")
    values_mul = torch.mul(weight_matrix,avg_matrix[0]) # 1. Hadamard Multiplication Elementwise Multiplication of matrices (the shape should remain the same)
    print("--sorted---")
    values_sort, index = torch.sort(values_mul, dim=2) # 2. sort the connections calculation results of each neuron from min to max and get the V2
    #values_sort, _ = values_mul.sort(0) 
    #print(values_sort)

    print("---Clipping---")
    min_quantile = torch.quantile(values_sort, 0.3).data.tolist()
    max_quantile = torch.quantile(values_sort, 0.6).data.tolist()
    max = values_sort >= max_quantile
    min = values_sort <= min_quantile
    
    values_sort[max] = 0
    values_sort[min] = 0
    values_clip = values_sort #torch.clamp(values_sort,min=min_quantile, max=max_quantile, out=None) # In this step the nd Tensor should get the top and bottom 30 percent of the values set to zero
    #print(values_clip)
    
    print("---Sum---")
    #values_sum = torch.sum(values_clip, dim=0) #Prediction Score
    values_sum = values_clip.sum(dim=0)
    #print(values_sum.size())    
    return values_sum, labels


#################
###   Train   ###
#################
def train_few_two_decide(skip=False): #model, num_epochs, random_seed, lr, momentum, train_loader, BATCH_SIZE, device ,skip = False):
    num_epochs=2
    lr = 0.0002
    momentum = 0.9
    w_decay = 0.001
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
    pred, labels = few_two_decide(model, trainloader)
    acc_train = accuracy_train(pred, labels)
    print("Train-Acc: ", acc_train)
    return 

# Train Accuaracy - F2D

def accuracy_train(pred, labels):
    count = 0
    for i in range(len(labels)):
        #print(pred[i].argmax(), labels[i])
        if (pred[i].argmax() == labels[i]):
            count += 1
    acc = 100*(count/len(labels))        
    return acc

#################
###    Test   ###
#################
def test_few_two_decide():
    model = upgraded_net_hook.ResNet20().to(device.device)
    model_point = torch.load("./utils/few2decide_model.tar", map_location=device.device)
    model.load_state_dict(model_point["state_dict"])

    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader.test_dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        pred, label = few_two_decide(model, dataloader.test_dataloader)
    
    print(f'Resnet - Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    acc_test = accuracy_train(pred, label)
    print("Test-Acc: ",acc_test)
    #return pred, label

print ("Train:")
train_few_two_decide(False)

print("Test:")
test_few_two_decide()