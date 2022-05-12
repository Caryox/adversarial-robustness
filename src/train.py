import sys
sys.path.append('./utils')

import param, dataloader

import basic_classification_nn

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F




#https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118
#from torch.autograd import Variable



def train(num_epochs, Net, loaders):
    
    Net.train()
        
    # Train the model
    total_step = len(loaders)
    optimizer = optim.SGD(Net.parameters(), lr = param.learning_rate, momentum=param.momentum)   
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(loaders,0):
            
            inputs, labels = data
            
            # clear gradients for this training step   
            optimizer.zero_grad() 

            output = Net(inputs)            
            loss = nn.CrossEntropyLoss(output, labels)                  
            
            # backpropagation, compute gradients 
            loss.backward()
            
            # apply gradients             
            optimizer.step()          
            
            running_loss += loss.item()
            if i % 100 == 99:
                #print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Loss: {:.3f}' 
                #       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), running_loss))
                
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, running_loss))            
                
                running_loss = 0.0
                
                
                #loss_list.append(loss.item())
                #train_counter.append(i)
    
    print("Training abgeschlossen")
    
train(param.num_epochs, basic_classification_nn.basic_Net, dataloader.train_dataloader)

