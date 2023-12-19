import os

from torchvision.datasets import FashionMNIST
from torchvision.datasets import MNIST

import cifar10.model_loader
import FashionMNIST.model_loader
import MNIST.model_loader

def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    elif dataset == 'FashionMNIST':
        net = FashionMNIST.model_loader.load(model_name, model_file, data_parallel)
    elif dataset == 'MNIST':
        net = MNIST.model_loader.load(model_name, model_file, data_parallel)
    return net
