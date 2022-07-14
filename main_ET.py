''' Issues 12
Dataloader
Basic NN
Training
Testing
Evaluation
'''
import utils
import src
from src import funtions as ff 
from src import Models as M
from src import results as pretrained
from utils import param 
from utils import dataloader as dtl
#from src import APEGAN #@Daniel: rename APE-GAN to APEGAN

############################################
# Loading Device
device = utils.device()

# Loading param: use param directly from utils.param
# Link https://github.com/Caryox/adversial-robustness/blob/master/utils/param.py


# Loading Data: all of our dataloaders can be found at https://github.com/Caryox/adversial-robustness/blob/main/utils/dataloader.py
# Transforms are already in dataloader modul

#MNIST
dataloader_train, dataloader_validation, dataloader_test = dtl.train_dataloader, dtl.validation_dataloader, dtl.test_dataloader
#dataloader(dataset, BATCH_SIZE, split_aufteilung, display_informations=False, num_of_worker=param.num_of_worker, random_seed=1337) #Kimmy: random_seed austauschen?

#CIFAR10: tobe done after MNIST

############################################
# Loading NN
#@Daniel&Kimmy: APEGAN has its own main file. How to handle APEGAN here?

basis_nn= M.upgraded_net
ensemble_nn1= M.upgraded_net_ensemble1
ensemble_nn2= M.upgraded_net_ensemble2

#APEGAN
#APEGAN=APEGAN.APEGAN_main

############################################
# We wanted to train, test and validate each of the model separately then compair the results with ensemble's result
# Train, Test, Validation - basis_nn
train_basis_nn = src.train(basis_nn, param.random_seed, param.learning_rate, param.momentum,
            param.num_epochs, dtl.train_dataloader,  param.BATCH_SIZE, device.device)
test_basis_nn = src.test(basis_nn, param.random_seed, dtl.test_dataloader, device.device)

validattion_basis_nn = src.vali(basis_nn, param.random_seed,
            param.num_epochs, dtl.validation_dataloader,  param.BATCH_SIZE, device.device)

# Train, Test, Validation - ensemble_nn1
train_basis_e1 = src.train(ensemble_nn1, param.random_seed, param.learning_rate, param.momentum,
            param.num_epochs, dtl.train_dataloader,  param.BATCH_SIZE, device.device)
test_basis_e1 = src.test(ensemble_nn1, param.random_seed, dtl.test_dataloader, device.device)

validattion_basis_e1 = src.vali(ensemble_nn1, param.random_seed,
            param.num_epochs, dtl.validation_dataloader,  param.BATCH_SIZE, device.device)

# Train, Test, Validation - ensemble_nn2        
train_basis_e2 = src.train(ensemble_nn2, param.random_seed, param.learning_rate, param.momentum,
            param.num_epochs, dtl.train_dataloader,  param.BATCH_SIZE, device.device)
test_basis_e2 = src.test(ensemble_nn2, param.random_seed, dtl.test_dataloader, device.device)

validattion_basis_e2 = src.vali(ensemble_nn2, param.random_seed,
            param.num_epochs, dtl.validation_dataloader,  param.BATCH_SIZE, device.device)