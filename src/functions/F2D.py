import sys
from typing import final
from unicodedata import decimal

from zmq import EVENT_LISTENING

sys.path.append('./utils')
sys.path.append('././src/Models')
sys.path.append('./src/APE_GAN')
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
import torch.optim.lr_scheduler as lr_scheduler
from APEGANModels import Generator

def few_two_decide_v2(model, inputs): 
    model.flatten.register_forward_hook(upgraded_net_hook.get_pooling('flatten'))
    pred = model(inputs)

    avg_matrix = upgraded_net_hook.average_pooling['flatten'] # Flatten-Output from the Average-Pooling Layer
    # Get linear.weights 
    weights = model.linear.weight.data.clone().detach().requires_grad_(True) #get weight matrix from ResNet
    
    #("---Hadamard product---")    
    sum_tensor = torch.ones(32, 10).to(device.device)
    for i in range(len(avg_matrix)):
        values_mul = torch.mul(weights, avg_matrix[i])# 1. Hadamard Multiplication Elementwise Multiplication of matrices (the shape should remain the same)
        
        #("--sorted---")
        values_sort, index = torch.sort(values_mul, dim=-1) # 2. sort the connections calculation results of each neuron from min to max and get the V2

        #("---Clipping---")    
        clip_tensor = torch.ones(10,64).to(device.device)
        for j in range(len(values_sort)): 
            
            for idx in range(len(values_sort[j].T)):
                if idx <32 or idx > 53:
                    values_sort[j][idx] = 0 
                    
            clip_tensor[j] = values_sort[j] #3. Clipping the sorted neurons (set neurons = 0) 

        #("---Sum---")
        values_sum  = torch.sum(clip_tensor, dim=1) #Prediction Score

        sum_tensor[i] = values_sum
    return sum_tensor


def test_attack(testloader, device,model, attack, resnet_path, apegan_path, input_channel, eps):
    print("Testing " + attack + " for epsilon " + str(eps))
    test_loader = testloader
    model = model
    model_point = torch.load(resnet_path, map_location=device)
    model.load_state_dict(model_point["state_dict"])
    gan_point = torch.load(apegan_path)
    G = Generator(input_channel).to(device)
    G.load_state_dict(gan_point["generator"])
    model.eval(), G.eval()

    normal_acc, f2d_acc,apegan_normal_acc,apegan_f2d_normal_acc, normal_adv_acc,f2d_adv_acc,apegan_adv_acc, apegan_f2d_adv_acc, n = 0, 0, 0, 0, 0, 0, 0, 0, 0
    if(input_channel == 1):
        attack, fmodel = attack_and_eval.attack(model, attack, (0, 1))
    else:
        attack, fmodel = attack_and_eval.attack(model, attack, (0, 255))
    for input, label in tqdm(test_loader, total=len(test_loader), leave=False):
            input, label = Variable(input.to(device)), Variable(label.to(device))

            normal_pred = model(input)
            f2d_pred = few_two_decide_v2(model, input)
            apegan_normal_inputs = G(input)
            apegan_normal_pred = model(apegan_normal_inputs)
            apegan_f2d_normal_pred = few_two_decide_v2(model, apegan_normal_inputs)
            input_adv, input_2, success = attack(fmodel, input, label, epsilons=eps)
            normal_adv_pred = model(input_2)
            f2d_adv_pred = few_two_decide_v2(model, input_2)
            apegan_inputs = G(input_2)
            apegan_adv_pred = model(apegan_inputs)
            apegan_f2d_adv_pred = few_two_decide_v2(model, apegan_inputs)

            # Prediction - model combination
            _, normal_pred = torch.max(normal_pred.data, 1)
            __, f2d_pred_index = torch.max(f2d_pred.data, 1)
            __, apegan_normal_pred = torch.max(apegan_normal_pred.data, 1)
            __, apegan_f2d_normal_pred = torch.max(apegan_f2d_normal_pred.data, 1)
            _, normal_adv_pred = torch.max(normal_adv_pred.data, 1)
            __, f2d_adv_pred = torch.max(f2d_adv_pred.data, 1)
            __, apegan_adv_pred = torch.max(apegan_adv_pred.data, 1)
            _, apegan_f2d_adv_pred = torch.max(apegan_f2d_adv_pred.data, 1)

            # Accuarcys - model combinations
            n += label.size(0)
            normal_acc += (normal_pred == label).sum().item()
            f2d_acc += (f2d_pred_index == label).sum().item()
            apegan_normal_acc += (apegan_normal_pred == label).sum().item()
            apegan_f2d_normal_acc += (apegan_f2d_normal_pred == label).sum().item()
            normal_adv_acc += (normal_adv_pred == label).sum().item()
            f2d_adv_acc += (f2d_adv_pred == label).sum().item()
            apegan_adv_acc += (apegan_adv_pred == label).sum().item()
            apegan_f2d_adv_acc += (apegan_f2d_adv_pred == label).sum().item()

    correct_normal = normal_acc
    correct_f2d = f2d_acc
    correct_apegan_normal = apegan_normal_acc
    correct_apegan_f2d = apegan_f2d_normal_acc
    correct_normal_adv = normal_adv_acc
    correct_f2d_adv = f2d_adv_acc
    correct_apegan_adv = apegan_adv_acc
    correct_apegan_f2d_adv = apegan_f2d_adv_acc

    normal_acc = 100* normal_acc / n
    f2d_acc = 100* f2d_acc / n
    apegan_normal_acc = 100* apegan_normal_acc / n
    apegan_f2d_normal_acc = 100* apegan_f2d_normal_acc / n
    normal_adv_acc = 100* normal_adv_acc / n
    f2d_adv_acc = 100* f2d_adv_acc / n
    apegan_adv_acc = 100* apegan_adv_acc / n
    apegan_f2d_adv_acc = 100* apegan_f2d_adv_acc / n
    print(f'Resnet - Accuracy of the network on {n} normal test images with {correct_normal} correct predictions: {normal_acc : .2f} %')
    print(f'F2D - Accuracy of the network on {n} normal test images with {correct_f2d} correct predictions: {f2d_acc : .2f} %')
    print(f'APEGAN with ResNet - Accuracy of the network on {n} normal test images with {correct_apegan_normal} correct predictions: {apegan_normal_acc : .2f} %')
    print(f'APEGAN with F2D - Accuracy of the network on {n} normal test images with {correct_apegan_f2d} correct predictions: {apegan_f2d_normal_acc : .2f} %')
    print(f'Resnet - Accuracy of the network on {n} adversarial test images with {correct_normal_adv} correct predictions: {normal_adv_acc : .2f} %')
    print(f'F2D - Accuracy of the network on {n} adversarial test images with {correct_f2d_adv} correct predictions: {f2d_adv_acc : .2f} %')
    print(f'APEGAN with ResNet - Accuracy of the network on {n} adversarial test images with {correct_apegan_adv} correct predictions: {apegan_adv_acc : .2f} %')
    print(f'APEGAN with F2D - Accuracy of the network on {n} adversarial test images with {correct_apegan_f2d_adv} correct predictions: {apegan_f2d_adv_acc : .2f} %')
