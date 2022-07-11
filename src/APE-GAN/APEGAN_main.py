import generate_adversarial_examples
import APEGAN_train
import APEGAN_test

import sys
sys.path.append('./utils')
sys.path.append('./src/Models')
               
import dataloader
import device

def main(train_loader, test_loader, device, input_channel=1, classes=10, eps=0.15):
    sample = next(iter(train_loader))
    input_channel = sample[0][0].shape[0]
    #generate_adversarial_examples.gen_adv(train_loader , test_loader , device, input_channel, classes)

    #APEGAN_train.APEGAN_Train(input_channel)
    attack = ["FGSM", "PGD", "L2DeepFool", "C&W"]
    for i in range(len(attack)):
        APEGAN_test.APEGAN_Test("./checkpoint/test/2.tar", "./src/APE-GAN/cnn.tar", test_loader, device, attack[i], input_channel, classes, eps)
    #APEGAN_test.APEGAN_Test("./checkpoint/test/2.tar", "./src/APE-GAN/cnn.tar",  test_loader, device, attack, input_channel, classes)

#main(dataloader.train_dataloader , dataloader.test_dataloader, device.device )
main(dataloader.train_dataloader_CIFAR10 , dataloader.test_dataloader_CIFAR10, device.device )