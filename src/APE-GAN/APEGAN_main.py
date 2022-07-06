import generate_adversarial_examples
import APEGAN_train
import APEGAN_test

import sys
sys.path.append('./utils')
sys.path.append('./src/Models')
               
import dataloader
import device

def main(input_channel=1, classes=10, eps=0.15):
    generate_adversarial_examples.gen_adv(dataloader.train_dataloader , dataloader.test_dataloader , device.device, input_channel, classes)

    APEGAN_train.APEGAN_Train(input_channel)

    APEGAN_test.APEGAN_Test("./checkpoint/test/2.tar", "./src/APE-GAN/cnn.tar",  dataloader.test_dataloader, device.device, input_channel, classes)

main()