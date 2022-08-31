#Import libraries
import generate_adversarial_examples
import APEGAN_train
import APEGAN_test
import ResNet_train
import sys

#Append needed function/module paths
sys.path.append('./utils')
sys.path.append('./src/Models')
sys.path.append('./src/functions')

#Import custom functions
import upgraded_net_hook  
import dataloader
import device
import F2D
import os

def main(train, train_loader, test_loader, device, resnet_path, apegan_path, resnet_epochs, apegan_epochs, eps):
    #Evaluate channel size by analyze sample from loader
    sample = next(iter(train_loader))
    input_channel = sample[0][0].shape[0]
    #Use corresponding resnet
    model = upgraded_net_hook.ResNet44(input_channel).to(device)
    if train:
        #Train resnet with data and gerneate adversarial samples
        ResNet_train.resnet_train(train_loader, device, model, resnet_path, resnet_epochs)
        generate_adversarial_examples.gen_adv(train_loader , device, model, resnet_path, eps, input_channel)
        APEGAN_train.APEGAN_Train(input_channel, apegan_epochs, apegan_path)
    attack = ["FGSM", "PGD", "L2DeepFool", "C&W"]
    #Run attacks
    for i in range(len(attack)):
        if(attack[i] in ["L2DeepFool", "LinfDeepFool", "C&W"]):
            F2D.test_attack(test_loader, device, model, attack[i], resnet_path, apegan_path, input_channel, None)
        else:
            for j in [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]:
                F2D.test_attack(test_loader, device, model, attack[i], resnet_path, apegan_path, input_channel, j)
#Run main for MNIST
print("Load data for MNIST and train model if no trained data is available (first run)")
main(True, dataloader.train_dataloader , dataloader.test_dataloader, device.device, "./utils/resnet_model_MNIST.tar", "./utils/apegan_model_MNIST.tar", 100, 2, 0.15)
#Run main for CIFAR-10
print("Load data for CIFAR-10 and train model if no trained data is available (first run)")
main(True, dataloader.train_dataloader_CIFAR10 , dataloader.test_dataloader_CIFAR10, device.device, "./utils/resnet_model_CIFAR10.tar", "./utils/apegan_model_CIFAR10.tar", 100, 30, 0.15)
