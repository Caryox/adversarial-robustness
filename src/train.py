# Custom Packages
import sys
sys.path.append('./utils')

import param, dataloader

from functions import basic_nn as basic

# Standard Packages
import torch
from torch import optim
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter
import torch.optim as optim

#https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118
from torch.autograd import Variable

def train(num_epochs, basic, BATCH_SIZE):
  basic.train()
  optimizer = torch.optim.SGD(params=basic.parameters(), lr=param.learning_rate, momentum=param.momentum)
  loss_func = nn.CrossEntropyLoss()
  total_step = len(dataloader.train_dataloader)

  for epoch in range(num_epochs):
    running_loss = 0
    for i, data in enumerate(dataloader.train_dataloader,0):
      inputs, labels = data
             
      optimizer.zero_grad() 

      output = basic(inputs)                               
      loss = loss_func(output, labels)                  
              
              # backpropagation, Berechnung des Gradienten
      loss.backward()    
              # Anwenden eines Optimierungsschrittes           
      optimizer.step()          
      running_loss += loss.item()
            
            #Ausgabe des Loss für einen Batch (=100) der jeweiligen Epoche
      if i % BATCH_SIZE == (BATCH_SIZE-1):
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'.format(epoch + 1, num_epochs, i + 1, total_step, running_loss))            
                
        running_loss = 0.0
    
  print("Training abgeschlossen")


train(param.num_epochs, basic.basic_Net, param.BATCH_SIZE)