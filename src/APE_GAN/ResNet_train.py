# Custom Packages
from math import gamma
import sys
from tkinter import Variable
sys.path.append('./utils')
sys.path.append('./src/Models')
import attack_and_eval               
import dataloader
import device
import foolbox
import upgraded_net_hook
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from tqdm import tqdm


def resnet_train(trainloader, device, model, resnet_path, epochs):
    lr = 0.01
    momentum = 0.9
    w_decay = 0.001
    milestones= [50,75]
    gamma = 0.1
    model = model
    model.apply(upgraded_net_hook.weights_init_uniform)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
    learningrate_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) # calc new learning rate
    loss_func = nn.CrossEntropyLoss().to(device)
    
    print("Training model for adversarial examples...")
    for epoch in range(epochs):
        print("Epoch: ", epoch + 1)
        model.train()
        
        for input, label in tqdm(trainloader, total=len(trainloader), leave=False):
            input, label = Variable(input.to(device)), Variable(label.to(device))
            
            pred = model(input)
            loss = loss_func(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        learningrate_scheduler.step()
    torch.save({"state_dict": model.state_dict()}, resnet_path)