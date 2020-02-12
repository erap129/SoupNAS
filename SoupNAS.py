import ast
import random
from functools import partial

import torchvision.datasets as datasets
from torch.nn import BatchNorm2d, Dropout, Conv2d, ModuleDict
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
from torch import nn
from skorch import NeuralNetClassifier

from PytorchModules import Conv2dAuto

fmnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(fmnist_train, batch_size=4, shuffle=True)
testloader = torch.utils.data.DataLoader(fmnist_test, batch_size=4, shuffle=False)
x_train = fmnist_train.train_data[:, None, :, :].float()
y_train = fmnist_train.train_labels.long()

layer_stock = {Conv2dAuto, BatchNorm2d, nn.ReLU, Dropout}


class SoupLayer:
    def __init__(self, layer, receptors):
        self.layer = layer
        self.receptors = receptors


def get_soup_layer(layer_type, input_channels, output_channels):
    if layer_type == Conv2dAuto:
        return Conv2dAuto(in_channels=input_channels, out_channels=output_channels, kernel_size=3, bias=False)
    elif layer_type == BatchNorm2d:
        return BatchNorm2d(input_channels)
    elif layer_type == Dropout:
        return Dropout(0.5)
    else:
        return layer_type()


class SoupNetwork(nn.Module):
    def __init__(self, layer_stock, soup_size, n_blocks, block_size):
        super(SoupNetwork, self).__init__()
        self.layer_stock = layer_stock
        self.soup_size = soup_size
        self.avg_pool = nn.AdaptiveAvgPool2d((10, 1))
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.soup = self.make_soup()
        self.linear = nn.Linear((x_train.shape[2] / block_size) * (x_train.shape[3] / block_size) * (x_train.shape[1] * block_size), 10)

    def make_soup(self):
        soup = ModuleDict()
        prev_block_channels = x_train.shape[1]
        for block_idx in range(self.n_blocks):
            first_in_block = True
            block_channels = x_train.shape[1] * (block_idx + 1)
            for i in range(self.block_size):
                random_layer = random.sample(layer_stock, 1)[0]
                if first_in_block and prev_block_channels is not None and random_layer == Conv2dAuto:
                    soup[f'{block_idx}_{i}_{[i]}'] = get_soup_layer(random_layer, prev_block_channels, block_channels)
                    first_in_block = False
                elif first_in_block:
                    soup[f'{block_idx}_{i}_{[i]}'] = get_soup_layer(random_layer, prev_block_channels, prev_block_channels)
                else:
                    soup[f'{block_idx}_{i}_{[i]}'] = get_soup_layer(random_layer, block_channels, block_channels)
            prev_block_channels = block_channels
        return soup

    def forward(self, X):
        for block_idx in range(self.n_blocks):
            for step in range(self.block_size):
                relevant_layers = [self.soup[k] for k in self.soup.keys() if (block_idx == int(k.split('_')[0]) and step in ast.literal_eval(k.split('_')[2]))]
                chosen_layer = random.choice(relevant_layers)
                X = chosen_layer(X)
            X = nn.MaxPool2d(2, stride=2)(X)
        # X = self.avg_pool(X).squeeze()
        X = X.view(X.size(0), -1)
        X = self.linear(X)
        X = nn.functional.softmax(X)
        return X


skorch_sn = NeuralNetClassifier(
    module=SoupNetwork,
    module__layer_stock=layer_stock,
    module__soup_size=30,
    module__n_blocks=3,
    module__block_size=10,
    max_epochs=100,
    lr=0.1,
    iterator_train__shuffle=True,
    device='cuda'
)


skorch_sn.fit(fmnist_train.train_data[:, None, :, :].float(), fmnist_train.train_labels.long())



# sn = SoupNetwork(layer_stock, 30)
# for batch in trainloader:
#     res = sn.forward(batch[0])
#     print(res)
