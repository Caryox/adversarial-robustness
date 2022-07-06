import sys
sys.path.append('./utils')
sys.path.append('./src/Models')
import dataloader
import device
import foolbox
import upgraded_net
import torch
from torch.autograd import Variable
from tqdm import tqdm
from APEGANModels import Generator


def APEGAN_Test(gan_path, model_path,  testloader, device, input_channel=1, classes=10, eps=0.15):
    test_loader = testloader

    model_point = torch.load(model_path)
    gan_point = torch.load(gan_path)
    model = upgraded_net.simple_net_upgraded(input_channel, classes).to(device)
    model.load_state_dict(model_point["state_dict"])

    G = Generator(input_channel).to(device)
    G.load_state_dict(gan_point["generator"])

    attack = foolbox.attacks.FGSM()
    fmodel = foolbox.models.PyTorchModel(model, bounds=(-1, 1), device=device)
    model.eval(), G.eval()
    normal_acc, adv_acc, ape_acc, n = 0, 0, 0, 0
    print("Start testing...")
    for input, label in tqdm(test_loader, total=len(test_loader), leave=False):
        input, label = Variable(input.to(device)), Variable(label.to(device))

        normal_acc = foolbox.utils.accuracy(fmodel, input, label) * 100

        input_adv, _, success = attack(fmodel, input, label, epsilons=eps)

        adv_acc = foolbox.utils.accuracy(fmodel, input_adv, label) * 100

        input_ape = G(input_adv)

        ape_acc = foolbox.utils.accuracy(fmodel, input_ape, label) * 100
        n += label.size(0)
    print("Accuracy: normal {:.6f}, fgsm {:.6f}, ape {:.6f}".format(
        normal_acc / n * 100,
        adv_acc / n * 100,
        ape_acc / n * 100))

#APEGAN_Test("./checkpoint/test/2.tar", "./src/APE-GAN/cnn.tar", 1, basic_nn.basic_Net,  dataloader.test_dataloader, device.device)