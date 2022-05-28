# train == train.py but with class ensemble
# https://github.com/ansh941/MnistSimpleCNN/blob/master/code/test.py
# paper https://arxiv.org/pdf/2008.10400.pdf

import torch  
import numpy as np
from utils import dataloader as dtl
from utils import param 
from src import Models as m
import functions
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loader -----------------------------------------------------------------#
#Using train_dataset_MNIST instead
#test_dataset = MnistDataset(training=False, transform=None)
test_dataloader = torch.utils.data.DataLoader(dtl.test_dataset_MNIST, batch_size=param.BATCH_SIZE, shuffle=False)

# model selection -------------------------------------------------------------#
model1 = m.basis_nn_k3
#model1 = functions.basis_nn --kernel size = 5
#model1 = m.basis_nn_k7
#model1 = m.basis_nn_k9

#@Bene: Can we actually use this? I remember Ivan talked about loading and saving models. Need your help here. Thanks :-)
#model1.load_state_dict(torch.load("../logs/%s/model%03d.pth"%(p_logdir,p_seed)))

#Using eval to check if model1 is a legit object?
model1.eval()

test_loss = 0
correct = 0
wrong_images = []
with torch.no_grad():
    for batch_idx, (img, label) in enumerate(test_dataloader):
        img, label = img.to(device), label.to(device)
        output = model1(img)
        test_loss += F.nll_loss(output, label, reduction='sum').item()
        #https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html
        #@Bene: I dont really get the idea of using nll_loss here. Should we keep it? Should we use nn.CrossEntropyLoss()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()
        wrong_images.extend(np.nonzero(~pred.eq(label.view_as(pred)).cpu().numpy())[0]+(100*batch_idx))

#Saving parameters as text?
#np.savetxt("../logs/%s/wrong%03d.txt"%(p_logdir,p_seed), wrong_images, fmt="%d")
#print(len(wrong_images), wrong_images)
