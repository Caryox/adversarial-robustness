# Reference: https://github.com/pytorch/examples/blob/main/mnist_hogwild/train.py

# Custom Packages
import sys
sys.path.append('./utils')
import dataloader
import param
import device
from functions import basic_nn as basic


import train
# Standard Packages

from torch.autograd import Variable
import torch.optim as optim
import torch.nn.parameter
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch import optim
import torch

print("test")

class test:
    def test(model, random_seed, test_loader, device):  #device
        torch.manual_seed(random_seed)
        test.test_epoch(model, test_loader, device)

        
    def test_epoch(model, test_loader, device):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data).to(device)
                test_loss += nn.CrossEntropyLoss(output, target.to(device), reduction='sum').item() # sum up batch loss
                pred = output.max(1)[1] # get the index of the max log-probability
                correct += pred.eq(target.to(device)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

test.test(basic.basic_Net, param.random_seed, dataloader.test_dataloader, device.device)
