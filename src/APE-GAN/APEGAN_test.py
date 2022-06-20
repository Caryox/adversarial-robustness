import sys
sys.path.append('./utils')
sys.path.append('./src/functions')
import dataloader
import device
import foolbox
import basic_nn
import torch
from torch.autograd import Variable
from tqdm import tqdm
from APEGANModels import Generator

def accuracy(y, t):
    pred = y.data.max(1, keepdim=True)[1]
    acc = pred.eq(t.data.view_as(pred)).cpu().sum()
    return acc


def APEGAN_Test(gan_path, model_path, channels, model, testloader, device):
    eps = 0.15
    test_loader = testloader

    model_point = torch.load(model_path)
    gan_point = torch.load(gan_path)

    #basic_model = model()

    #model = basic_model().to(device)
    model.load_state_dict(model_point["state_dict"])

    G = Generator(channels).to(device)
    G.load_state_dict(gan_point["generator"])
    #loss_cre = nn.CrossEntropyLoss().to(device)
    attack = foolbox.attacks.FGSM()
    fmodel = foolbox.models.PyTorchModel(model, bounds=(-1, 1), device=device)
    model.eval(), G.eval()
    normal_acc, adv_acc, ape_acc, n = 0, 0, 0, 0
    for input, label in tqdm(test_loader, total=len(test_loader), leave=False):
        input, label = Variable(input.to(device)), Variable(label.to(device))

        pred = model(input)
        normal_acc += accuracy(pred, label)

        input_adv, _, success = attack(fmodel, input, label, epsilons=eps)
        pred_adv = model(input_adv)
        adv_acc += accuracy(pred_adv, label)

        input_ape = G(input_adv)
        pred_ape = model(input_ape)
        ape_acc += accuracy(pred_ape, label)
        n += label.size(0)
    print("Accuracy: normal {:.6f}, fgsm {:.6f}, ape {:.6f}".format(
        normal_acc / n * 100,
        adv_acc / n * 100,
        ape_acc / n * 100))

APEGAN_Test("./checkpoint/test/2.tar", "./src/APE-GAN/cnn.tar", 1, basic_nn.basic_Net,  dataloader.test_dataloader, device.device)