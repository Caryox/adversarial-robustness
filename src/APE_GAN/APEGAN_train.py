# APE-GAN train class

# Import custom Module
import sys
sys.path.append('./.')
from utils.device import device
from APEGANModels import Generator, Discriminator

# -*- coding: utf-8 -*-

import os
import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset

from tqdm import tqdm
import matplotlib.pyplot as plt

# function - show images
def show_images(e, x, x_adv, x_fake, save_dir):
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    for i in range(5):
        axes[0, i].axis("off"), axes[1, i].axis("off"), axes[2, i].axis("off")
        axes[0, i].imshow(x[i].cpu().numpy().transpose((1, 2, 0)))
        # axes[0, i].imshow(x[i, 0].cpu().numpy(), cmap="gray")
        axes[0, i].set_title("Normal")

        axes[1, i].imshow(x_adv[i].cpu().numpy().transpose((1, 2, 0)))
        # axes[1, i].imshow(x_adv[i, 0].cpu().numpy(), cmap="gray")
        axes[1, i].set_title("Adv")

        axes[2, i].imshow(x_fake[i].cpu().numpy().transpose((1, 2, 0)))
        # axes[2, i].imshow(x_fake[i, 0].cpu().numpy(), cmap="gray")
        axes[2, i].set_title("APE-GAN")
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, "result_{}.png".format(e)))

def APEGAN_Train(input_channel, epochs, apegan_path):
    lr = 0.0002
    batch_size = 128
    xi1, xi2 = 0.7, 0.3
    gen_epochs = 2

    # load normal and adv data
    train_data = torch.load("./src/APE_GAN/data.tar")
    x_tmp = train_data["normal"][:5]
    input_adv_tmp = train_data["adv"][:5]

    train_data = TensorDataset(train_data["normal"], train_data["adv"])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    G = Generator(input_channel).to(device) # Generator Output
    D = Discriminator(input_channel).to(device) # Discriminator Output

    #optimizer for Generator and Discriminator
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # loss functions
    loss_bce = nn.BCELoss()
    loss_mse = nn.MSELoss()
    # train model and save weights
    print_str = "\t".join(["{}"] + ["{:.6f}"] * 2)
    print("\t".join(["{:}"] * 3).format("Epoch", "Gen_Loss", "Dis_Loss"))
    for epoch in range(epochs):
        #G.eval()
        #input_fake = G(Variable(input_adv_tmp.to(device))).data
        #show_images(epoch, x_tmp, input_adv_tmp, input_fake, check_path)
        G.train()
        gen_loss, dis_loss, n = 0, 0, 0
        for i, data in enumerate(train_loader,0):
        #for input, input_adv in tqdm(train_loader, total=len(train_loader), leave=False):
            input, input_adv = data
            current_size = input.size(0)
            input, input_adv = Variable(input.to(device)), Variable(input_adv.to(device))
            
            # Train Discriminator
            label_real = Variable(torch.ones(current_size).to(device))
            label_fake = Variable(torch.zeros(current_size).to(device))

            pred_real = D(input).squeeze()
            input_fake = G(input_adv)
            pred_fake = D(input_fake).squeeze()

            loss_D = loss_bce(pred_real, label_real) + loss_bce(pred_fake, label_fake)
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Train Generator
            for _ in range(gen_epochs):
                input_fake = G(input_adv)
                pred_fake = D(input_fake).squeeze()

                loss_G = xi1 * loss_mse(input_fake, input) + xi2 * loss_bce(pred_fake, label_real)
                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()
            
            # Calc losses
            gen_loss += loss_D.item() * input.size(0)
            dis_loss += loss_G.item() * input.size(0)
            n += input.size(0)
        print(print_str.format(epoch, gen_loss / n, dis_loss / n))
        G.eval()
        input_fake = G(Variable(input_adv_tmp.to(device))).data
        show_images(epoch, x_tmp, input_adv_tmp, input_fake, "./src/APE_GAN/Images/")
        G.train()
    
    torch.save({"generator": G.state_dict(), "discriminator": D.state_dict()}, apegan_path)

    
