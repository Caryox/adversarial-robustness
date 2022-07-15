import sys
from typing import final
from unicodedata import decimal

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

def few_two_decide_v2(model, inputs): 
    model.linear.register_forward_hook(upgraded_net_hook.get_activation('linear'))
    model.flatten.register_forward_hook(upgraded_net_hook.get_pooling('flatten'))
    pred = model(inputs)

    #print("Get matrix")
    weight_matrix = upgraded_net_hook.activation['linear']
    avg_matrix = upgraded_net_hook.average_pooling['flatten']
    #avg_matrix = avg_matrix.mean(dim=(-2, -1))
    
    #print(weight_matrix.size())
    #print(avg_matrix.size())
    
    #print("---Hadamard product---")
    #values_mul = torch.mul(weight_matrix.T, avg_matrix) # 1. Hadamard Multiplication Elementwise Multiplication of matrices (the shape should remain the same)
    
    sum_tensor = torch.ones(32, 10).to(device.device)
    for i in range(len(avg_matrix)):
        values_mul = torch.mul(weight_matrix, avg_matrix[i])# 1. Hadamard Multiplication Elementwise Multiplication of matrices (the shape should remain the same)
        #print(values_mul.size())
    
        #print("--sorted---")
        values_sort, index = torch.sort(values_mul, dim=-1) # 2. sort the connections calculation results of each neuron from min to max and get the V2
        #print(values_sort.size()) #.detach().numpy().round(2))

        #print("---Clipping---")
        #ToDO - Clipping auf richtigkeit prüfen
        clip_tensor = torch.ones(10,64).to(device.device)
        for j in range(len(values_sort)): 
            
            min_quantile = torch.quantile(values_sort[j], 1/3).item()
            max_quantile = torch.quantile(values_sort[j], 2/3).item()
            
            #print(min_quantile, max_quantile)

            value_clip = torch.where(values_sort[j] > min_quantile, values_sort[j], 0)
            value_clip =torch.where(value_clip < max_quantile, value_clip, 0)
            clip_tensor[j] = value_clip
        #print("---Sum---")
        values_sum  = torch.sum(clip_tensor, dim=1) #Prediction Score
        #print(values_sum)

        sum_tensor[i] = values_sum
    return sum_tensor


"""def few_two_decide(model, dataloader):
    #model.eval()
    
    model.linear.register_forward_hook(upgraded_net_hook.get_activation('linear'))
    model.flatten.register_forward_hook(upgraded_net_hook.get_pooling('flatten'))
    sample = next(iter(dataloader))
    inputs, labels = sample
    inputs, labels = Variable(inputs.to(device.device)), Variable(labels.to(device.device))
    pred = model(inputs)

    print("Get matrix")
    weight_matrix = upgraded_net_hook.activation['linear']
    avg_matrix = upgraded_net_hook.average_pooling['flatten']
    #avg_matrix = avg_matrix.mean(dim=(-2, -1))
    
    #print(weight_matrix.size())
    #print(avg_matrix.size())
    
    #print("---Hadamard product---")
    #values_mul = torch.mul(weight_matrix.T, avg_matrix) # 1. Hadamard Multiplication Elementwise Multiplication of matrices (the shape should remain the same)
    
    sum_tensor = torch.ones(32, 10)
    for i in range(len(avg_matrix)):
        values_mul = torch.mul(weight_matrix, avg_matrix[i])# 1. Hadamard Multiplication Elementwise Multiplication of matrices (the shape should remain the same)
        #print(values_mul.size())
    
        #print("--sorted---")
        values_sort, index = torch.sort(values_mul, dim=-1) # 2. sort the connections calculation results of each neuron from min to max and get the V2
        #print(values_sort.size()) #.detach().numpy().round(2))

        print("---Clipping---")
        #ToDO - Clipping auf richtigkeit prüfen
        clip_tensor = torch.ones(10,64)
        for j in range(len(values_sort)): 
            
            min_quantile = torch.quantile(values_sort[j], 1/3).item()
            max_quantile = torch.quantile(values_sort[j], 2/3).item()
            
            print(min_quantile, max_quantile)

            value_clip = torch.where(values_sort[j] > min_quantile, values_sort[j], 0)
            value_clip =torch.where(value_clip < max_quantile, value_clip, 0)
            clip_tensor[j] = value_clip
        print("---Sum---")
        values_sum  = torch.sum(clip_tensor, dim=1) #Prediction Score
        print(values_sum)

        sum_tensor[i] = values_sum   
    return sum_tensor, labels"""


#################
###   Train   ###
#################
def train_few_two_decide_v2(skip=False): #model, num_epochs, random_seed, lr, momentum, train_loader, BATCH_SIZE, device ,skip = False):
    num_epochs=2
    lr = 0.0002
    momentum = 0.9
    w_decay = 0.001
    model = upgraded_net_hook.ResNet44().to(device.device)
    #model.apply(upgraded_net_hook.weights_init_uniform)
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
         
            # placeholder for batch features
            for i, data in enumerate(trainloader):
                inputs, labels = data
                inputs, labels = Variable(inputs.to(device.device)), Variable(labels.to(device.device))
                f2d = few_two_decide_v2(model, inputs)
                #print(f2d[0])
                pred = model(inputs)
                #print(pred[0])
                loss = loss_func(pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred = pred.data.max(1, keepdim=True)[1]
        torch.save({"state_dict": model.state_dict()}, "./utils/few2decide_model.tar")
        #pred, labels = few_two_decide(model, trainloader)
        #print(labels.size())
        acc_train = accuracy_train(pred, labels)
        print("Train-Acc: ", acc_train)
        return pred, labels

"""def train_few_two_decide(skip=False): #model, num_epochs, random_seed, lr, momentum, train_loader, BATCH_SIZE, device ,skip = False):
    num_epochs=5
    lr = 0.0002
    momentum = 0.9
    w_decay = 0.001
    model = upgraded_net_hook.ResNet44().to(device.device)
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
         
            # placeholder for batch features
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
        #print(labels.size())
        acc_train = accuracy_train(pred, labels)
    print("Train-Acc: ", acc_train)
    return pred, labels"""

# Train Accuaracy - F2D

def accuracy_train(pred, labels):
    count = 0
    for i in range(len(labels)):
        #print(pred[i])
        #print(pred[i].argmax(), labels[i])
        if (pred[i].argmax() == labels[i]):
            count += 1
    acc = 100*(count/len(labels))        
    return acc

#################
###    Test   ###
#################
def test_few_two_decide():
    model = upgraded_net_hook.ResNet44().to(device.device)
    model_point = torch.load("./utils/few2decide_model.tar", map_location=device.device)
    model.load_state_dict(model_point["state_dict"])
    model.eval()
    correct = 0
    total = 0
    f2d = 0
    with torch.no_grad():
        for data in dataloader.test_dataloader:
            images, labels = data
            images, labels = Variable(images.to(device.device)), Variable(labels.to(device.device))
            outputs = model(images)
            print(outputs[0])
            pred = few_two_decide_v2(model, images)
            print(pred[0])
            _, predicted = torch.max(outputs.data, 1)
            __, f2d_pred = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(pred.shape)
            f2d += (f2d_pred == labels).sum().item()
    print(f'Resnet - Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    print(f'F2D - Accuracy of the network on the 10000 test images: {100 * f2d // total} %')
    #acc_test = accuracy_train(pred, label)
    #print("Test-Acc: ",acc_test)
    #return pred, label




"""def test_few_two_decide():
    model = upgraded_net_hook.ResNet44().to(device.device)
    model_point = torch.load("./utils/few2decide_model.tar", map_location=device.device)
    model.load_state_dict(model_point["state_dict"])

    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader.test_dataloader):
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        pred, label = few_two_decide(model, dataloader.test_dataloader)
    #print(f'Resnet - Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    acc_test = accuracy_train(pred, label)
    print("Test-Acc: ",acc_test)
    return pred, label"""

print ("Train:")
train_few_two_decide_v2(False)
#print(pred, labels)

print("Test:")
test_few_two_decide()
