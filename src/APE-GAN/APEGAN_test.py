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
import attack_and_eval

def APEGAN_Test(gan_path, model_path,  testloader, device, attack_name="FGSM", input_channel=1, classes=10, eps=0.15):
    test_loader = testloader

    model_point = torch.load(model_path)
    gan_point = torch.load(gan_path)
    model = upgraded_net.simple_net_upgraded(input_channel, classes).to(device)
    model.load_state_dict(model_point["state_dict"])

    G = Generator(input_channel).to(device)
    G.load_state_dict(gan_point["generator"])
    model.eval(), G.eval()
    attack, fmodel = attack_and_eval.attack(model, attack_name)
    
    normal_acc, adv_acc, ape_acc, n = 0, 0, 0, 0
    print("Start testing...")
    for input, label in tqdm(test_loader, total=len(test_loader), leave=False):
        input, label = Variable(input.to(device)), Variable(label.to(device))

        normal_acc = attack_and_eval.evaluation(fmodel, input, label)

        input_adv, _, success = attack(fmodel, input, label, epsilons=eps)

        adv_acc = attack_and_eval.evaluation(fmodel, input_adv, label)

        input_ape = G(input_adv)

        ape_acc = attack_and_eval.evaluation(fmodel, input_ape, label)
        n += label.size(0)
    print("Accuracy: normal {:.6f}".format(
        normal_acc / n * 100))
    print("Accuracy: " + str(attack_name) + " {:.6f}".format(
        adv_acc / n * 100))
    print("Accuracy: APEGAN {:.6f}".format(
        ape_acc / n * 100))