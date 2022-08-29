import torch
import torchvision
import torchvision.transforms as transform
batch_size_test=32
batch_size_train=32
random_seed=1337

def dataloader_ens(dataset):

	if (str(dataset) == "MNIST"):
		train_loader = torch.utils.data.DataLoader(
			torchvision.datasets.MNIST('../../data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
			batch_size=batch_size_train, shuffle=True, drop_last=True,worker_init_fn=random_seed)

		test_loader = torch.utils.data.DataLoader(
			torchvision.datasets.MNIST('../../data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                             ])),
			batch_size=batch_size_test, shuffle=True,drop_last=True,worker_init_fn=random_seed)
		print("Loaded MNIST dataloader")
	
	if (str(dataset) == "CIFAR"):
		train_loader = torch.utils.data.DataLoader(
			torchvision.datasets.CIFAR10('../../data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                              torchvision.transforms.Normalize(
                                 (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                                torchvision.transforms.Resize(size=(28,28))
                             ])),
			batch_size=batch_size_train, shuffle=True, drop_last=True,worker_init_fn=random_seed)

		test_loader = torch.utils.data.DataLoader(
			torchvision.datasets.CIFAR10('../../data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                                torchvision.transforms.Resize(size=(28,28))
                             ])),
			batch_size=batch_size_test, shuffle=True,drop_last=True,worker_init_fn=random_seed)
		print("Loaded CIFAR-10 dataloader")
	return (train_loader, test_loader)