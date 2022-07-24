import generate_adversarial_examples
import APEGAN_train
import APEGAN_test

import sys
sys.path.append('./utils')
sys.path.append('./src/Models')
sys.path.append('./src/functions')
import upgraded_net_hook  
import dataloader
import device
import F2D

def main(train_loader, test_loader, device, resnet_path, apegan_path, epochs, eps=0.3):
    sample = next(iter(train_loader))
    input_channel = sample[0][0].shape[0]
    model = upgraded_net_hook.ResNet44(input_channel).to(device)
    generate_adversarial_examples.gen_adv(train_loader , device, model, resnet_path, epochs, eps)

    APEGAN_train.APEGAN_Train(input_channel, epochs, apegan_path)
    attack = ["FGSM"]
    for i in range(len(attack)):
        F2D.test_attack(test_loader, device, model, attack[i], resnet_path, apegan_path, input_channel, eps)
    
main(dataloader.train_dataloader , dataloader.test_dataloader, device.device, "./utils/resnet_model_100_MNIST.tar", "./utils/apegan_model_100_MNIST.tar", 100)
#main(dataloader.train_dataloader_CIFAR10 , dataloader.test_dataloader_CIFAR10, device.device, "./utils/resnet_model_100_CIFAR10.tar", "./utils/apegan_model_100_CIFAR10.tar", 100)
