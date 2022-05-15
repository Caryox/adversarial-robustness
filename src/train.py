# Custom Packages
import sys
sys.path.append('./utils')
import dataloader
import param
from functions import basic_nn as basic
# Standard Packages

from torch.autograd import Variable
import torch.optim as optim
import torch.nn.parameter
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch import optim
import torch


# https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118


def train(num_epochs, basic, BATCH_SIZE):
    basic.train()
    loss_func = nn.CrossEntropyLoss()
    total_step = len(dataloader.train_dataloader)
    optimizer_SGD = torch.optim.SGD(
        basic.parameters(), lr=param.learning_rate, momentum=param.momentum)

    for epoch in range(num_epochs):
        running_loss = 0
        for i, data in enumerate(dataloader.train_dataloader, 0):
            
            optimizer_SGD.zero_grad()

            inputs, labels = data

            output = basic(inputs)
            loss = loss_func(output, labels)

            # backpropagation, Berechnung des Gradienten
            loss.backward()
            # Anwenden eines Optimierungsschrittes
            optimizer_SGD.step()
            running_loss += loss.item()

            # Ausgabe des Loss für einen Batch (=100) der jeweiligen Epoche
            if i % BATCH_SIZE == (BATCH_SIZE-1):
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'.format(epoch +
                      1, num_epochs, i + 1, total_step, running_loss))

            running_loss = 0.0

    print("Training abgeschlossen")
    #Speichern des Models
    Path = '.mnis_net.pth'
    torch.save(basic.state_dict(), Path)


#train(param.num_epochs, basic.basic_Net, param.BATCH_SIZE)
