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


def gen_adv(trainloader, device, model, resnet_path, eps, input_channel):
    model = model
    model_point = torch.load(resnet_path)
    model.load_state_dict(model_point["state_dict"])
    normal_data, adv_data = None, None
    model.eval()
    if(input_channel == 1):
        attack, fmodel = attack_and_eval.attack(model, "FGSM", (0, 1))
    else:
        attack, fmodel = attack_and_eval.attack(model, "FGSM", (0, 255)) 
    print("Generating adversarial examples...")
    for input, label in tqdm(trainloader, total=len(trainloader), leave=False):
        input, label = Variable(input.to(device)), Variable(label.to(device))
        raw_adv_input, clipped_adv_input, success = attack(fmodel, input, label, epsilons=eps)
        input, raw_adv_input = input.data, raw_adv_input.data
        if normal_data is None:
            normal_data, adv_data = input, raw_adv_input
        else:
            normal_data = torch.cat((normal_data, input))
            adv_data = torch.cat((adv_data, raw_adv_input))

    torch.save({"normal": normal_data, "adv": adv_data}, "./src/APE_GAN/data.tar")
