# Reference: https://github.com/Dianevera/heart-prediction/blob/94f2b9919d78a47ef3ee4d711e879ab1d08b273c/heartpredictions/LSTM/create_dataloaders.py

# StandardPackage

import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt


# CustomPackages
import param

def dataloader(dataset, BATCH_SIZE, split_aufteilung, display_informations=False, num_of_worker=param.num_of_worker, random_seed=1337):
    torch.backends.cudnn.enabled = True
    torch.manual_seed(random_seed)
    
    lengths = [round(len(dataset) * split) for split in split_aufteilung] # calculate lengths per dataset without consideration Split_Aufteilung

    r = 0
    
    for i in range(len(lengths)):
        r_tmp = lengths[i] % 3 # Value an der Stelle i modulo 3
        lengths[i] = lengths[i] - r_tmp 
        r += r_tmp
        print(r)
    lengths[2] += r
    
    #Calculation of the dataset-sizes
    train = torch.utils.data.Subset(dataset, range(0, lengths[0]))
    validation = torch.utils.data.Subset(dataset, range(lengths[0], lengths[0] + lengths[1]))
    test = torch.utils.data.Subset(dataset, range(lengths[0] + lengths[1], lengths[0] + lengths[1] + lengths[2]))
    
    # train loader
    train_dataloader = torch.utils.data.DataLoader(
        train,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        worker_init_fn=random_seed,
        num_workers=0, #num_Worker = 0 because MemoryError 
        #prefetch_factor=1,
        persistent_workers=False,
        pin_memory=True
    )
    
    # validation loader
    validation_dataloader = torch.utils.data.DataLoader(
        validation,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        worker_init_fn=random_seed,
        num_workers=0,
        #prefetch_factor=1,
        persistent_workers=False,
        pin_memory=True
    )

    # test loader
    test_dataloader = torch.utils.data.DataLoader(
        test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        worker_init_fn=random_seed,
        num_workers=0,
        #prefetch_factor=1,
        persistent_workers=False,
        pin_memory=True
    )
    
    # 
    if display_informations:
        print(f'Total dataset: {len(train_dataloader) + len(validation_dataloader) + len(test_dataloader)}, '
            f'train dataset: {len(train_dataloader)}, val dataset: {len(validation_dataloader)}, test_dataset: {len(test_dataloader)}')
    
    return train_dataloader, validation_dataloader, test_dataloader


#MNIST
    
train_dataset_MNIST = torchvision.datasets.MNIST('./data', train=True, download=True,
                             transform = transforms.Compose([
                                            transforms.Resize(param.IMAGE_SIZE),
                                            transforms.CenterCrop(param.IMAGE_SIZE),
                                            transforms.ToTensor(),
                                            #transforms.Normalize(*param.NORM),
                                            ]
                                            ))
                                
test_dataset_MNIST = datasets.MNIST(root='./data', train=False, download=True, 
                                    transform = transforms.Compose([
                                            transforms.Resize(param.IMAGE_SIZE),
                                            transforms.CenterCrop(param.IMAGE_SIZE),
                                            transforms.ToTensor(),
                                            #transforms.Normalize(*param.NORM),
                                            ]))

#CIFAR-10
train_dataset_CIFAR10 = torchvision.datasets.CIFAR10('./data', train=True, download=True,
                                transform = transforms.Compose([
                                            transforms.Resize(param.IMAGE_SIZE),
                                            transforms.CenterCrop(param.IMAGE_SIZE),
                                            transforms.ToTensor(),
                                            #transforms.Normalize(*param.NORM),
                                            ]))

test_dataset_CIFAR10 = datasets.CIFAR10(root='./data', train=False, download=True, 
                                    transform = transforms.Compose([
                                            transforms.Resize(param.IMAGE_SIZE),
                                            transforms.CenterCrop(param.IMAGE_SIZE),
                                            transforms.ToTensor(),
                                            #transforms.Normalize(*param.NORM),
                                            ]))
#Concat Train and Test Dataset --> Whole Data 
'''MNIST'''
dataset_MNIST = ConcatDataset([train_dataset_MNIST, test_dataset_MNIST])

'''CIFAR-10'''
dataset_CIFAR10 = ConcatDataset([train_dataset_CIFAR10, test_dataset_CIFAR10])

#Custom Train, Validation, Test Split

'''MNIST'''
train_dataloader, validation_dataloader, test_dataloader = dataloader(dataset_MNIST, param.BATCH_SIZE, param.SPLIT_AUFTEILUNG, param.num_of_worker)

'''CIFAR-10'''
train_dataloader_CIFAR10, validation_dataloader_CIFAR10, test_dataloader_CIFAR10 = dataloader(dataset_CIFAR10, param.BATCH_SIZE, param.SPLIT_AUFTEILUNG, param.num_of_worker)

