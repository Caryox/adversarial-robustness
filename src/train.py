# Reference: https://github.com/pytorch/examples/blob/main/mnist_hogwild/train.py

# Custom Packages
import torch
from torch import optim
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter
import torch.optim as optim
from torch.autograd import Variable
from functions import basic_nn as basic
import device
import param
import dataloader
import sys
sys.path.append('./utils')
# Standard Packages


# https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118

print('train')


class train:
    def train(model, random_seed, lr, momentum, num_epochs, train_loader, BATCH_SIZE, device):
        torch.manual_seed(random_seed)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        # for epoch in range(num_epochs):
        train.train_epoch(num_epochs, model, train_loader,
                          optimizer, BATCH_SIZE, device)

    def train_epoch(num_epochs, model, train_loader, optimizer, BATCH_SIZE, device):  
        model.train()
        loss_func = nn.CrossEntropyLoss()
        total_step = len(train_loader)

        for epoch in range(num_epochs):
            running_loss = 0
            for i, data in enumerate(train_loader, 0):

                optimizer.zero_grad()

                inputs, labels = data

                output = model(inputs).to(device)
                loss = loss_func(output, labels).to(device)

                # backpropagation, Berechnung des Gradienten
                loss.backward()
                # Anwenden eines Optimierungsschrittes
                optimizer.step()
                running_loss += loss.item()

                if i % BATCH_SIZE == (BATCH_SIZE-1):
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'.format(epoch +
                                                                             1, num_epochs, i + 1, total_step, running_loss))

                    running_loss = 0.0

            print("Training abgeschlossen")
        
        # Speichern des Models
        Path = '.mnis_net.pth'
        torch.save(model.state_dict(), Path)


train.train(basic.basic_Net, param.random_seed, param.learning_rate, param.momentum,
            param.num_epochs, dataloader.train_dataloader,  param.BATCH_SIZE, device.device)
