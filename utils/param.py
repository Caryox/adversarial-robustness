"""File for params for ARGAN/Few2Decide"""

###################################
############## MNIST ##############
###################################

num_epochs = 20
BATCH_SIZE = 32
IMAGE_SIZE = 28
learning_rate = 0.01
momentum = 0.5
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
train_counter = []
num_of_worker = 1
random_seed = 1337


# Train, Validation, Test
TEST_SPLIT = 0.2
#TEST_SPLIT = 0.00166666666666666666666666666667 # Test for C&W
VALID_SPLIT = 0.1
TRAIN_SPLIT = (1-(TEST_SPLIT+VALID_SPLIT))

SPLIT_AUFTEILUNG = {TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT}

#Its recommended to further use CIFAR 10 Normalizations