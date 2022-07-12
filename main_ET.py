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

############################################
# Loading Device
device = utils.device()

# Loading param: use param directly from utils.param
# Link https://github.com/Caryox/adversial-robustness/blob/master/utils/param.py


# Loading Data: all of our dataloaders can be found at https://github.com/Caryox/adversial-robustness/blob/main/utils/dataloader.py
# Using transfrom from torchvision because can not find our transforms function.
# Tobe replaced later
import torchvision.transforms as transforms 
transform = transforms.Compose([
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,)),
])

#MNIST
#@Kimmy: Am I abusing your functions? 
dataloader_test = dtl.test_dataloader()
dataloader_train = dtl.train_dataloader()
dataloader_validation = dtl.validation_dataloader()

#CIFAR10: tobe done after MNIST

############################################
# Loading NN
#@Daniel&Kimmy: APEGAN has its own main file. How to handle APEGAN here?

basis_nn= M.upgraded_net()
ensemble_nn1= M.upgraded_net_ensemble1()
ensemble_nn2= M.upgraded_net_ensemble2()

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