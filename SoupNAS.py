import random
from functools import partial

import torchvision.datasets as datasets
from torch.nn import BatchNorm2d
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
from torch import nn

from PytorchModules import Conv2dAuto

fmnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(fmnist_train, batch_size=4, shuffle=True)
testloader = torch.utils.data.DataLoader(fmnist_test, batch_size=4, shuffle=False)

print(fmnist_train.train_data[0].shape)
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)
layer_stock = {conv3x3, BatchNorm2d, F.relu}


class SoupNetwork(nn.Module):
    def __init__(self, layer_stock, soup_size):
        super(SoupNetwork, self).__init__()
        self.layer_stock = layer_stock
        self.soup_size = soup_size
        self.soup = self.make_soup()

    def make_soup(self):
        soup = {}
        for i in range(soup_size):
            random_layer = random.choice(layer_stock)()
            soup.add(random_layer)

    def forward(self, X):



NUM_LAYERS = 30

for i in range(NUM_LAYERS):
    pass


conv = conv3x3(in_channels=1, out_channels=64)
conv(fmnist_train.train_data[:10][:, None, :, :].float())