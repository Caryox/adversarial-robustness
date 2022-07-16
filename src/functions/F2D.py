import sys
from typing import final
from unicodedata import decimal

sys.path.append('./utils')
sys.path.append('././src/Models')
import attack_and_eval  
import upgraded_net_hook
import param
import dataloader
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import device
from torch.autograd import Variable
from tqdm import tqdm
#import torch.optim.lr_scheduler as lr_scheduler

def few_two_decide_v2(model, inputs): 
    model.linear.register_forward_hook(upgraded_net_hook.get_activation('linear')) 
    model.flatten.register_forward_hook(upgraded_net_hook.get_pooling('flatten'))
    pred = model(inputs)

    #print("Get matrix")
    weight_matrix = upgraded_net_hook.activation['linear']  # Weight-Matrix of the last FCL
    avg_matrix = upgraded_net_hook.average_pooling['flatten'] # Flatten-Output from the Average-Pooling Layer
    
    #print(weight_matrix.size())
    #print(avg_matrix.size())
    
    #print("---Hadamard product---")    
    sum_tensor = torch.ones(32, 10).to(device.device)
    for i in range(len(avg_matrix)):
        values_mul = torch.mul(weight_matrix, avg_matrix[i])# 1. Hadamard Multiplication Elementwise Multiplication of matrices (the shape should remain the same)
        
        #print("--sorted---")
        values_sort, index = torch.sort(values_mul, dim=-1) # 2. sort the connections calculation results of each neuron from min to max and get the V2

        #print("---Clipping---")    
        clip_tensor = torch.ones(10,64).to(device.device) 
        for j in range(len(values_sort)): 
            
            min_quantile = torch.quantile(values_sort[j], 1/3).item() # lower bound
            max_quantile = torch.quantile(values_sort[j], 2/3).item() # upper bound
            
            #print(min_quantile, max_quantile)

            value_clip = torch.where(values_sort[j] > min_quantile, values_sort[j], 0)
            value_clip =torch.where(value_clip < max_quantile, value_clip, 0)
            clip_tensor[j] = value_clip #3. Clipping the sorted neurons (set neurons = 0) 
        
        #print("---Sum---")
        values_sum  = torch.sum(clip_tensor, dim=1) #Prediction Score

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
    num_epochs=30
    lr = 0.002
    momentum = 0.9
    w_decay = 0.001
    milestones= [1,2]
    gamma = 0.1

    model = upgraded_net_hook.ResNet44().to(device.device)
    model.apply(upgraded_net_hook.weights_init_uniform)
    trainloader = dataloader.train_dataloader
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
    #learningrate_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    loss_func = nn.CrossEntropyLoss().to(device.device)
    if skip:
        model_point = torch.load("./utils/few2decide_model.tar", map_location=device.device)
        model.load_state_dict(model_point["state_dict"])
    else:
        model.train()
        #acc = 0
        for epoch in range(num_epochs):
            print("Epoch: ", epoch)
         
            # placeholder for batch features
            #for i, data in enumerate(trainloader):
            for input, label in tqdm(trainloader, total=len(trainloader), leave=False):
                inputs, labels = Variable(input.to(device.device)), Variable(label.to(device.device))
                #pred = few_two_decide_v2(model, inputs)
                #print(pred)
                pred = model(inputs)
                #print(pred[0])
                loss = loss_func(pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred = pred.data.max(1, keepdim=True)[1]
                #acc += accuracy_train(pred, labels)
            #learningrate_scheduler.step()
        torch.save({"state_dict": model.state_dict()}, "./utils/few2decide_model.tar")
        #pred, labels = few_two_decide(model, trainloader)
        #print("Train-Acc: ", acc)
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
        for input, label in tqdm(dataloader.test_dataloader, total=len(dataloader.test_dataloader), leave=False):
            images, labels = Variable(input.to(device.device)), Variable(label.to(device.device))
            outputs = model(images)
            pred = few_two_decide_v2(model, images)
            _, predicted = torch.max(outputs.data, 1)
            __, f2d_pred = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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

def generate_adversarial_examples(model, data_loader, device, data_path, eps=0.15):
    normal_data, adv_data = None, None
    model = model
    model_point = torch.load("./utils/few2decide_model.tar", map_location=device)
    model.load_state_dict(model_point["state_dict"])
    model.eval()
    attacks, fmodel = attack_and_eval.attack(model, "FGSM")
    print("Generating adversarial examples...")
    for i, data in enumerate(data_loader,0):
        input, label = data
        input, label = Variable(input.to(device)), Variable(label.to(device))

        input_adv, _ = attacks(fmodel, input, label, epsilons=eps)
        
        input, input_adv = input.data, input_adv.data
        if normal_data is None:
            normal_data, adv_data = input, input_adv
        else:
            normal_data = torch.cat((normal_data, input))
            adv_data = torch.cat((adv_data, input_adv))
    torch.save({"normal": normal_data, "adv": adv_data}, data_path)


def test_attack(testloader, device, eps=0.15):
    test_loader = testloader
    model = upgraded_net_hook.ResNet44().to(device)
    model_point = torch.load("./utils/few2decide_model.tar", map_location=device)
    model.load_state_dict(model_point["state_dict"])
    model.eval()
    normal_acc, f2d_acc, adv_acc, normal_adv_acc,f2d_adv_acc, n = 0, 0, 0, 0, 0, 0
    attack, fmodel = attack_and_eval.attack(model, "FGSM", (-255, 255))
    for input, label in tqdm(test_loader, total=len(test_loader), leave=False):
            input, label = Variable(input.to(device)), Variable(label.to(device))

            normal_pred = model(input)
            f2d_pred = few_two_decide_v2(model, input)
            input_adv, _, success = attack(fmodel, input, label, epsilons=eps)
            normal_adv_pred = model(input_adv)
            f2d_adv_pred = few_two_decide_v2(model, input_adv)


            _, normal_pred = torch.max(normal_pred.data, 1)
            __, f2d_pred = torch.max(f2d_pred.data, 1)
            _, normal_adv_pred = torch.max(normal_adv_pred.data, 1)
            __, f2d_adv_pred = torch.max(f2d_adv_pred.data, 1)
            n += label.size(0)
            normal_acc += (normal_pred == label).sum().item()
            f2d_acc += (f2d_pred == label).sum().item()
            normal_adv_acc += (normal_adv_pred == label).sum().item()
            f2d_adv_acc += (f2d_adv_pred == label).sum().item()


    print(f'Resnet - Accuracy of the network on the normal test images: {100 * normal_acc // n} %')
    print(f'F2D - Accuracy of the network on the 10000 test images: {100 * f2d_acc // n} %')
    print(f'Resnet - Accuracy of the network on the adversarial test images: {100 * normal_adv_acc // n} %')
    print(f'F2D - Accuracy of the network on the adversarial test images: {100 * f2d_adv_acc // n} %')

print ("Train:")
#train_few_two_decide_v2(False)
#print(pred, labels)

print("Test:")
#test_few_two_decide()

#generate_adversarial_examples(upgraded_net_hook.ResNet44().to(device.device), dataloader.test_dataloader, device.device, "./utils/adv_data.tar")

test_attack(dataloader.test_dataloader, device.device, eps=0.15)