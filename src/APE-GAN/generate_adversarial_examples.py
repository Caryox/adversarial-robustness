# Custom Packages
from math import gamma
import sys
from tkinter import Variable
sys.path.append('./utils')
sys.path.append('./src/Models')
                
import dataloader
import device
import foolbox
import upgraded_net
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable



def gen_adv(trainloader, testloader, device, input_channel=1, classes=10):
    num_epochs=2
    lr = 0.0002
    momentum = 0.9
    w_decay = 0.001
    milestones= [50,75]
    gamma = 0.1
    model = upgraded_net.simple_net_upgraded(input_channel, classes).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
    learningrate_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) # calc new learning rate
    loss_func = nn.CrossEntropyLoss().to(device)
    
    print("Training model for adversarial examples...")
    for epoch in range(num_epochs):
        train_loss, train_acc, train_n = 0, 0, 0
        test_loss, test_acc, test_n = 0, 0, 0

        model.train()
        
        for i, data in enumerate(trainloader,0):
            input, label = data
            input, label = Variable(input.to(device)), Variable(label.to(device))
            
            pred = model(input)
            loss = loss_func(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * label.size(0)
            
            #accuarcy
            pred = pred.data.max(1, keepdim=True)[1]
            acc = pred.eq(label.data.view_as(pred)).cpu().sum()
            train_acc += acc
            train_n += label.size(0)
        
        model.eval()
        for i, data in enumerate(testloader,0):
            input, label = data
            input, label = Variable(input.to(device)), Variable(label.to(device))
            
            pred = model(input)
            test_loss += loss.item() * label.size(0)
            
            #acc
            pred = pred.data.max(1, keepdim=True)[1]
            acc = pred.eq(label.data.view_as(pred)).cpu().sum()
            test_acc += acc
            
            test_n += label.size(0)
        learningrate_scheduler.step()
        
        
    #Generate Adv. Examples
    eps = 0.15
    train_acc, adv_acc, train_n = 0, 0, 0
    normal_data, adv_data = None, None
    attack = foolbox.attacks.FGSM()
    fmodel = foolbox.models.PyTorchModel(model, bounds=(-1, 1), device=device)
    print("Generating adversarial examples...")
    for i, data in enumerate(trainloader,0):
        input, label = data
        input, label = Variable(input.to(device)), Variable(label.to(device))
        pred = model(input)
        #accuarcy
        pred = pred.data.max(1, keepdim=True)[1]
        #print(pred)
        
        acc = pred.eq(label.data.view_as(pred)).cpu().sum()
        train_acc += acc
        
        input_adv, _, success = attack(fmodel, input, label, epsilons=eps)
        pred_adv = model(input_adv)
        
        pred_adv = pred_adv.data.max(1, keepdim=True)[1]
        acc = pred_adv.eq(label.data.view_as(pred_adv)).cpu().sum()
        adv_acc += acc
        train_n += label.size(0)

        input, input_adv = input.data, input_adv.data
        if normal_data is None:
            normal_data, adv_data = input, input_adv
        else:
            normal_data = torch.cat((normal_data, input))
            adv_data = torch.cat((adv_data, input_adv))

    print("Accuracy(normal) {:.6f}, Accuracy(FGSM) {:.6f}".format(train_acc / train_n * 100, adv_acc / train_n * 100))
    torch.save({"normal": normal_data, "adv": adv_data}, "./src/APE-GAN/data.tar")
    torch.save({"state_dict": model.state_dict()}, "./src/APE-GAN/cnn.tar")




#gen_adv(basic_nn.basic_Net, dataloader.train_dataloader , dataloader.test_dataloader , device.device)           
            
            
            
            
    
    