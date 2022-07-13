"""File for params"""

###################################
############## MNIST ##############
###################################

num_epochs = 20
BATCH_SIZE = 32
IMAGE_SIZE = 28
learning_rate = 0.01
momentum = 0.5
NORM = (0.1307,), (0.3081,)
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
train_counter = []
num_of_worker = 1
random_seed = 1337


# Train, Validation, Test
TEST_SPLIT = 0.2
VALID_SPLIT = 0.1
TRAIN_SPLIT = (1-(TEST_SPLIT+VALID_SPLIT))

SPLIT_AUFTEILUNG = {TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT}

# optimizer


###################################
############## CIFAR ##############
###################################


              
