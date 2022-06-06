"""Class to add/reverse gaussian noise to/
from images to generate Xr for ARGAN"""
import torch
import param
torch.manual_seed(param.random_seed)
# https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(
            self.mean, self.std
            )

# Does not work


class ReverseGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor - torch.randn(tensor.size()) * self.std + self.mean
