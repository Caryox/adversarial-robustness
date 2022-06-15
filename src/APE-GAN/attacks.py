#Reference: https://github.com/owruby/APE-GAN/blob/1095ac2f6c0cf85fc75fcd0802c8cce4762dbacf/utils.py

import torch
from torch.autograd import Variable


def fgsm(model, x, t, loss_func, eps, min=0, max=1):
    if not isinstance(x, Variable):
        x, t = Variable(x.cuda(), requires_grad=True), Variable(t.cuda())
    x.requires_grad = True
    y = model(x)
    loss = loss_func(y, t)
    model.zero_grad()
    loss.backward(retain_graph=True)

    return Variable(torch.clamp(x.data + eps * torch.sign(x.grad.data), min=min, max=max))