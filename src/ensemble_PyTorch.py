#Plan C
#Sorry, dass ich 3 mal Ensemble geändert habe :-( ich fühle mich mit den anderen 2 nicht ok
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms #if not using our folder. Pls delete this import when cleaning done
from torchensemble.utils.logging import set_logger
from torchensemble import VotingClassifier  # voting is a classic ensemble strategy
import utils #our folder
import notepads
# Load data from utils folder
train_loader = utils.dataloader.train_dataloader()
test_loader = utils.dataloader.train_dataloader()

'''
exemple with MNIST - NOT UNSEREN FOLDER STRUCTURE
# Load MNIST dataset
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

train = datasets.MNIST('../Dataset', train=True, download=True, transform=transform)
test = datasets.MNIST('../Dataset', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=True)
'''

# Define the ensemble
ensemble = VotingClassifier(
    estimator=notepads.simple_cnn(),        #We want our basemodel here as estimator. 
    n_estimators=1,                        # number of base estimators. It can be more than 1. But we only have 1 :)))
)
# Set the criterion
criterion = nn.CrossEntropyLoss()           # training objective
ensemble.set_criterion(criterion)

# Set the optimizer
learning_rate=0.01 #as in param.py learning_rate
weight_decay=5e-4  #as in tutorial

ensemble.set_optimizer(
    "Adam",                                 # type of parameter optimizer
    lr=learning_rate,                       # learning rate of parameter optimizer
    weight_decay=weight_decay,              # weight decay of parameter optimizer
)

# Set the learning rate scheduler -WURDE RAUS KOMMENTIERT, WEIL ICH DIE NUTZUNG NICHT 100% VERSTEHEN KANN
#ensemble.set_scheduler(
#    "CosineAnnealingLR",                    # type of learning rate scheduler
#    T_max=epochs,                           # additional arguments on the scheduler
#)

# Train the ensemble
epochs = 50 #is it good enough?
ensemble.fit(
    train_loader,
    epochs=epochs,                          
)

# Evaluate the ensemble
acc = ensemble.predict(test_loader)         # testing accuracy