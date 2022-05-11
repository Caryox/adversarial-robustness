"""File for params"""

###################################
############## MNIST ##############
###################################

num_epochs = 20
BATCH_SIZE = 64
IMAGE_SIZE = 32
#learning_rate = 0.01
#momentum = 0.5
NORM = (0.5,), (0.5,)
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
train_counter = []

# Train, Validation, Test
TEST_SPLIT = 0.2
VALID_SPLIT = 0
TRAIN_SPLIT = (1-(TEST_SPLIT+VALID_SPLIT))

SPLIT_AUFTEILUNG = {TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT}

###################################
############## CIFAR ##############
###################################


              
