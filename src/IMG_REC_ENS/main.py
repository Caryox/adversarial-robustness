
import os

import torch
import foolbox as fb
import sys
import torch.optim as optim
import time  # For sleep command
sys.path.append('../functions')
sys.path.append('../Models')
import ensemble_cnn
from test_ensemble import test
from train_ensemble import train
from evaluate_accuracy_image_rectification_ensemble import evaluate_attack
from evaluate_adversarial_detection import evaluate_adversarial_detection
from calculate_threshold_t import calculate_threshold_t
from generate_adversarial_samples_ens import generate_pertubated_examples
from dataloader_ens import dataloader_ens
from img_median_smoothing import median_smoothing
from img_bit_reduction import bit_reduction
from clean_nn_weights import reset_weights


###Support function for main###

def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


# Setting runtime params
random_seed = 1337
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(random_seed)  # Static random seed for reproducibility
device = torch.device("cpu")

def core_main():

    print("1. Use MNIST dataset")
    print("2. Use CIFAR-10 dataset")
    print("3. Return to defense selecion")
    user_input = int(input("Selection: "))

    # match-support comes with python 3.10
    if user_input == 1:
        dataset_name = "MNIST"
    elif user_input == 2:
        dataset_name = "CIFAR"
    elif user_input == 3:
        exit()
    train_loader, test_loader = dataloader_ens(dataset_name)

    if (str(dataset_name) == "MNIST"):
        network = ensemble_cnn.ensemble_rectification(1, 10)
        network.apply(reset_weights)
    else:
        network = ensemble_cnn.ensemble_rectification(3, 10)
        network.apply(reset_weights)

    print("#####################")

    print("1. Train Model")
    print("2. Load Model")
    user_input = int(input("Selection: "))

    if user_input == 1:
        optimizer = optim.SGD(network.parameters(), lr=0.007, momentum=0.5)
        print("Epochs? (default 10)")
        n_epochs = int(input("Epochs: "))
        print("1. Normal train")
        print("2. PGF+normal train")
        print("3. PGF+FGSM train")
        user_input = int(input("Train strategy: "))
        if user_input == 1:
            train(network, optimizer, train_loader, dataset_name, n_epochs)
        elif user_input == 2:
            train(network, optimizer, train_loader,
              dataset_name, n_epochs, adversarial_train=True)
        elif user_input == 3:
            train(network, optimizer, train_loader, dataset_name,
                n_epochs, adversarial_train=True, double_adv=True)

    if user_input == 2:
        optimizer = optim.SGD(network.parameters(), lr=0.007, momentum=0.5)
        print("1. Load normal trained data")
        print("2. Load PGD+normal trained data")
        print("3. Load PGD+FGSM trained data")
        user_input = int(input("Load model weights: "))
        if user_input == 1:
            if str(dataset_name) == "MNIST":
                network.load_state_dict(torch.load(
                   '../../src/results/model_integrated_ens.pth'))
                print('Loaded normal trained MNIST weights from file')
                optimizer.load_state_dict(torch.load(
                    '../../src/results/optimizer_integrated.pth'))
                print('Loaded normal trained MNIST optimizer from file')
            elif str(dataset_name) == "CIFAR":
                network.load_state_dict(torch.load(
                    '../../src/results/model_integrated_ens_cifar.pth'))
                print('Loaded normal trained CIFAR-10 weights from file')
                optimizer.load_state_dict(torch.load(
                    '../../src/results/optimizer_integrated_cifar.pth'))
                print('Loaded normal trained CIFAR-10 optimizer from file')
    ###PGD+normal###
    if user_input == 2:
        if str(dataset_name) == "MNIST":
            # ../src/results/model_integrated_ens_adv_mnist_03.pth
            network.load_state_dict(torch.load(
                '../../src/results/model_integrated_ens_adv.pth'))
            print('Loaded PGD+normal trained MNIST weights from file')
            # ../src/results/optimizer_integrated_adv_mnist_03.pth
            optimizer.load_state_dict(torch.load(
                '../../src/results/optimizer_integrated_adv.pth'))
            print('Loaded PGD+normal trained MNIST optimizer from file')
        elif str(dataset_name) == "CIFAR":
            network.load_state_dict(torch.load(
                '../../src/results/model_integrated_ens_adv_cifar.pth'))
            print('Loaded PGD+normal trained CIFAR-10 weights from file')
            optimizer.load_state_dict(torch.load(
                '../../src/results/optimizer_integrated_adv_cifar.pth'))
            print('Loaded PGD+normal trained CIFAR-10 optimizer from file')
    ###PGD#FGSM###
    if user_input == 3:
        if str(dataset_name) == "MNIST":
            # ../src/results/model_integrated_ens_adv_mnist_03.pth
            network.load_state_dict(torch.load(
                '../../src/results/model_integrated_ens_adv_mnist_03.pth'))
            print('Loaded PGD+FGSM trained MNIST weights from file')
            # ../src/results/optimizer_integrated_adv_mnist_03.pth
            optimizer.load_state_dict(torch.load(
                '../../src/results/optimizer_integrated_adv_mnist_03.pth'))
            print('Loaded PGD+FGSM trained MNIST optimizer from file')
        elif str(dataset_name) == "CIFAR":
            network.load_state_dict(torch.load(
                '../../src/results/model_integrated_ens_adv_cifar_03.pth'))
            print('Loaded PGD+FGSM trained CIFAR-10 weights from file')
            optimizer.load_state_dict(torch.load(
                '../../src/results/optimizer_integrated_adv_cifar_03.pth'))
            print('Loaded PGD+FGSM trained CIFAR-10 optimizer from file')

    print("Loaded/Trained Model succeessfully!")
    print("###################################")
    time.sleep(2)
    cls()
    print("Determinate Test-Accuracy")
    base_accuracy = test(network, test_loader)
    print("###################################")
    print("Attack evaluation")
    print("1. Linf-FGSM")
    print("2. L2-FGSM (FGM)")
    print("3. Linf-PGD")
    print("4. L2-PGD")
    print("5. L2-Deepfool")
    print("6. L2-Carlini Wagner")
    user_input = int(input("Attack: "))
    print("###################################")
    print("Set epsilon value (default 0.1,0.5 or 1)")
    epsilon = float(input("Epsilon e: "))
    print("###################################")
    print("Code will do automatically: \n1.) Measure accuracy of classification methods against selected attack \n2.) Create adversarial sample set for estimating threshold t and evaluate adversarial detection capabilities")
    if user_input == 1:
        evaluate_attack(test_loader, network, dataset_name, base_accuracy, device,
            is_attack=True, attack=fb.attacks.FGSM(), epsilon=[epsilon], t=0, median_kernel=2)
        ad_data, ad_label, original_data, original_label = generate_pertubated_examples(
            network, test_loader, fb.attacks.FGSM(), dataset_name, n=200, epsilon=[epsilon])

    if user_input == 2:
        evaluate_attack(test_loader, network, dataset_name, base_accuracy, device,
            is_attack=True, attack=fb.attacks.FGM(), epsilon=[epsilon], t=0, median_kernel=2)
        ad_data, ad_label, original_data, original_label = generate_pertubated_examples(
            network, test_loader, fb.attacks.FGM(), dataset_name, n=200, epsilon=[epsilon])

    if user_input == 3:
        evaluate_attack(test_loader, network, dataset_name, base_accuracy, device,
            is_attack=True, attack=fb.attacks.PGD(), epsilon=[epsilon], t=0, median_kernel=2)
        ad_data, ad_label, original_data, original_label = generate_pertubated_examples(
            network, test_loader, fb.attacks.PGD(), dataset_name, n=200, epsilon=[epsilon])

    if user_input == 4:
        evaluate_attack(test_loader, network, dataset_name, base_accuracy, device,
            is_attack=True, attack=fb.attacks.L2PGD(), epsilon=[epsilon], t=0, median_kernel=2)
        ad_data, ad_label, original_data, original_label = generate_pertubated_examples(
            network, test_loader, fb.attacks.L2PGD(), dataset_name, n=200, epsilon=[epsilon])

    if user_input == 5:
        evaluate_attack(test_loader, network, dataset_name, base_accuracy, device, is_attack=True,
            attack=fb.attacks.deepfool.L2DeepFoolAttack(), epsilon=[epsilon], t=0, median_kernel=2)
        ad_data, ad_label, original_data, original_label = generate_pertubated_examples(
            network, test_loader, fb.attacks.deepfool.L2DeepFoolAttack(), dataset_name, n=200, epsilon=[epsilon])

    if user_input == 6:
        evaluate_attack(test_loader, network, dataset_name, base_accuracy, device, is_attack=True,
            attack=fb.attacks.carlini_wagner.L2CarliniWagnerAttack(), epsilon=[epsilon], t=0, median_kernel=2)
        ad_data, ad_label, original_data, original_label = generate_pertubated_examples(
            network, test_loader, fb.attacks.carlini_wagner.L2CarliniWagnerAttack(), dataset_name, n=20, epsilon=[epsilon])

    cut = int(min(len(ad_data), len(original_data))/2)
    print("Train/Test split after {} samples".format(cut))

    #! ToDO: Check if CW is used
    if dataset_name == "CIFAR":
        x = ad_data
        y = original_data
        train_data_adv = torch.empty(0, 3, 28, 28)
        train_data_orig = torch.empty(0, 3, 28, 28)

        for i in range(100):
            train_data_adv = torch.cat(
                (train_data_adv, x[i].reshape(1, 3, 28, 28)), dim=0)
        for i in range(100):
            train_data_orig = torch.cat(
                (train_data_orig, y[i].reshape(1, 3, 28, 28)), dim=0)
#! This
        data_train = torch.cat((train_data_adv, train_data_orig), dim=0)

########

        x = ad_data
        y = original_data
        test_data_adv = torch.empty(0, 3, 28, 28)
        test_data_orig = torch.empty(0, 3, 28, 28)
        for i in range(100, 200):
            test_data_adv = torch.cat(
                (test_data_adv, x[i].reshape(1, 3, 28, 28)), dim=0)
        for i in range(100, 200):
            test_data_orig = torch.cat(
                (test_data_orig, y[i].reshape(1, 3, 28, 28)), dim=0)
        data_test = torch.cat((test_data_adv, test_data_orig), dim=0)
        label_train = torch.cat((ad_label[0:cut], ad_label[0:cut]), dim=0)

        if (len(ad_data) <= len(original_data)):
            label_test = torch.cat(
                (ad_label[cut:len(ad_data)], ad_label[cut:len(ad_data)]), dim=0)
        else:
            label_test = torch.cat(
                (ad_label[cut:len(original_data)], ad_label[cut:len(original_data)]), dim=0)

    if dataset_name == "MNIST":
    # ! MNIST, CIFAR deactivate line
        data_train = torch.cat((ad_data[0:cut], original_data[0:cut]), dim=0)
        label_train = torch.cat((ad_label[0:cut], ad_label[0:cut]), dim=0)

        if (len(ad_data) <= len(original_data)):
            data_test = torch.cat(
                (ad_data[cut:len(ad_data)], original_data[cut:len(ad_data)]), dim=0)  # ! MNIST
            label_test = torch.cat(
                (ad_label[cut:len(ad_data)], ad_label[cut:len(ad_data)]), dim=0)
        else:
            data_test = torch.cat((ad_data[cut:len(
                original_data)], original_data[cut:len(original_data)]), dim=0)  # ! MNIST
            label_test = torch.cat(
                (ad_label[cut:len(original_data)], ad_label[cut:len(original_data)]), dim=0)

    differences, adversarial_map = evaluate_adversarial_detection(
        data_train, label_train, network, str(dataset_name))
    t_value = calculate_threshold_t(
        differences, adversarial_map, correction_t=1.0, print_output=True)
    evaluate_adversarial_detection(data_test, label_test, network, str(
        dataset_name), t_value, evaluate=True)
    print("###################################")
    print("###################################")
    core_main()
core_main()
