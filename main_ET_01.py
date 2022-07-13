''' Issues 12
Dataloader
Basic NN
Training
Testing
Evaluation
'''
import sys
sys.path.append('./utils')
sys.path.append('./src')
sys.path.append('./src/Models')
import Models as M
import param 
import dataloader as dtl
import device 
import train
import test
import validation
#from upgraded_net import simple_net_upgraded
############################################
# Loading Device
device = device.get_default_device()

# Loading param: use param directly from utils.param

# Transforms are already in dataloader modul

#MNIST
dataloader_train, dataloader_validation, dataloader_test = dtl.train_dataloader, dtl.validation_dataloader, dtl.test_dataloader
#dataloader(dataset, BATCH_SIZE, split_aufteilung, display_informations=False, num_of_worker=param.num_of_worker, random_seed=1337) #Kimmy: random_seed austauschen?

#CIFAR10: tobe done after MNIST

############################################
# Loading NN
#@Daniel&Kimmy: APEGAN has its own main file. How to handle APEGAN here?

basis_nn= M.simple_net_upgraded(1,10)
#Ensemble of 3 NN models <- to be done

#APEGAN
#APEGAN=APEGAN.APEGAN_main

############################################

# Train, Test, Validation - basis_nn
train_basis_nn = train.train(basis_nn, param.random_seed, param.learning_rate, param.momentum,
            param.num_epochs, dtl.train_dataloader,  param.BATCH_SIZE, device.device)
test_basis_nn = test.test(basis_nn, param.random_seed, dtl.test_dataloader, device.device)

validattion_basis_nn = validation.val.vali(basis_nn, param.random_seed,
            param.num_epochs, dtl.validation_dataloader,  param.BATCH_SIZE, device.device)

#Train, Test and Validation of Ensemble NN: tbd