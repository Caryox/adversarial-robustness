import sys
from utils import RandomNoiseXr
from utils import param
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

sys.path.append('../utils')
torch.manual_seed(param.random_seed)

"""
where p(·) is a conversion function which adds the random noise to
the input data X, and p−1(·) is an inverse function of p(·),
which restores the input data X transformed by p(·).
"""

dataset = datasets.MNIST('./data', train=True, download=True,
                         transform=transforms.Compose([transforms.Resize(
                             param.IMAGE_SIZE),
                             transforms.CenterCrop(param.IMAGE_SIZE),
                             transforms.ToTensor(),
                             transforms.Normalize(*param.NORM),
                             RandomNoiseXr.AddGaussianNoise(0., 1.)
                             ]
                            )
                         )


loader = DataLoader(
    dataset,
    batch_size=param.BATCH_SIZE,
    shuffle=False,
    drop_last=True,
    worker_init_fn=param.random_seed,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True
)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu"
    )
real_batch = next(iter(loader))
plt.figure(figsize=(16, 16))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],
           padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.show()
