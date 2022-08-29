# Custom Packages
import sys
sys.path.append('./utils')
import dataloader
import param
import device
from functions import basic_nn as basic
# Standard Packages

from torch.autograd import Variable
import torch.nn.parameter
import torch.nn as nn
import torchvision
import torch


print('validation')
class val:
    def vali(model, random_seed, num_epochs, validation_loader, BATCH_SIZE, device):
        torch.manual_seed(random_seed)
        val.vali_epochs(num_epochs, model, validation_loader, BATCH_SIZE, device)

    def vali_epochs(num_epochs, model, validation_loader, BATCH_SIZE, device): 
        loss_func = nn.CrossEntropyLoss()
        total_step = len(validation_loader)
        
        for epoch in range(num_epochs):
            validation_loss = 0

            for i, data in enumerate(validation_loader,0): 
                input, labels = data
                # don't need to create a computation graph --> don't create a backwardstep for the validation set
                with torch.no_grad():
                    outputs_validation= model(input)
                    validation_loss=loss_func(outputs_validation, labels).to(device)
                    assert validation_loss.requires_grad == False
                    
                # We accumulate the test loss
                validation_loss += validation_loss.item()
                    
                if i % BATCH_SIZE == (BATCH_SIZE-1):
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'.format(epoch +
                                                                             1, num_epochs, i + 1, total_step,  validation_loss))
                    validation_loss = 0.0

val.vali(basic.basic_Net, param.random_seed,
            param.num_epochs, dataloader.validation_dataloader,  param.BATCH_SIZE, device.device)
