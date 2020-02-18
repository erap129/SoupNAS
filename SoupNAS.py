import ast
import random
from functools import partial

import torchvision.datasets as datasets
from torch.nn import BatchNorm2d, Dropout, Conv2d, ModuleDict, ModuleList
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
from skorch.callbacks import TensorBoard, Callback
from torch import nn
from skorch import NeuralNetClassifier
from PytorchModules import Conv2dAuto, ResidualBlock


writer = SummaryWriter()
fmnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(fmnist_train, batch_size=4, shuffle=True)
testloader = torch.utils.data.DataLoader(fmnist_test, batch_size=4, shuffle=False)
x_train = fmnist_train.train_data[:, None, :, :].float()
y_train = fmnist_train.train_labels.long()

layer_stock = {ResidualBlock, Conv2dAuto, BatchNorm2d, nn.ReLU, Dropout}


class MyTensorBoard(TensorBoard):
    def __init__(self, *args, X, **kwargs):
        self.X = X
        super().__init__(*args, **kwargs)

    def add_graph(self, module, X):
        """"Add a graph to tensorboard

        This requires to run the module with a sample from the
        dataset.

        """
        self.writer.add_graph(module, X.cuda())

    def on_batch_begin(self, net, X, **kwargs):
        if self.first_batch_:
            # only add graph on very first batch
            # self.add_graph(net.module_, X)
            pass

    def on_epoch_end(self, net, **kwargs):
        self.add_graph(net.module_, self.X)
        print(net.module_.current_layers)

class ShuffleOrder(Callback):
    def on_epoch_end(self, net, **kwargs):
        net.module_.shuffle_net()


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
    elif layer_type == ResidualBlock:
        return ResidualBlock(input_channels)
    else:
        return layer_type()


class SoupNetwork(nn.Module):
    def __init__(self, layer_stock, soup_size, n_blocks, block_size):
        super(SoupNetwork, self).__init__()
        self.layer_stock = layer_stock
        self.soup_size = soup_size
        self.n_blocks = n_blocks
        self.block_size = block_size
        # self.expansion_convs = ModuleList([Conv2dAuto(in_channels=x_train.shape[1] * i, out_channels=x_train.shape[1] * (i + 1), kernel_size=3) for i in range(1, n_blocks+1)])
        self.expansion_convs = ModuleList([Conv2dAuto(in_channels=max(1, 32 * i), out_channels=32 * (i + 1), kernel_size=3) for i in range(n_blocks)])
        self.soup = self.make_linear_soup()
        self.shuffle_net()
        self.linear = nn.Linear(int(x_train.shape[2] / 2 ** n_blocks) * int(x_train.shape[3] / 2 ** n_blocks) * int(32 * (n_blocks)), 10)


    def make_linear_soup(self):
        soup = ModuleDict()
        for block_idx in range(self.n_blocks):
            block_channels = max(1, 32*block_idx)
            for i in range(self.block_size):
                random_layer = random.sample(layer_stock, 1)[0]
                # soup[f'{block_idx}_{i}_{[i]}'] = get_soup_layer(random_layer, block_channels, block_channels)
                soup[f'{block_idx}_{i}_{[i for i in range(self.block_size)]}'] = get_soup_layer(random_layer,
                                                                                                block_channels,
                                                                                                block_channels)
        return soup

    def shuffle_net(self):
        self.current_layers = ModuleDict()
        for block_idx in range(self.n_blocks):
            self.current_layers[str(block_idx)] = ModuleList()
            for step in range(self.block_size):
                relevant_layers = [self.soup[k] for k in self.soup.keys() if
                                   (block_idx == int(k.split('_')[0]) and step in ast.literal_eval(k.split('_')[2]))]
                self.current_layers[str(block_idx)].append(random.choice(relevant_layers))

    def forward(self, X):
        for block_idx in range(self.n_blocks):
            # for step in range(self.block_size):
            #     relevant_layers = [self.soup[k] for k in self.soup.keys() if (block_idx == int(k.split('_')[0]) and step in ast.literal_eval(k.split('_')[2]))]
            #     chosen_layer = random.choice(relevant_layers)
            for layer in self.current_layers[str(block_idx)]:
                X = layer(X)
            X = self.expansion_convs[block_idx](X)
            X = nn.MaxPool2d(2, stride=2)(X)
        X = X.view(X.size(0), -1)
        X = self.linear(X)
        X = nn.functional.softmax(X)
        return X

callbacks = []
writer = SummaryWriter()
callbacks.append(MyTensorBoard(writer, X=fmnist_train.train_data[:, None, :, :].float()[:2]))
callbacks.append(ShuffleOrder())

skorch_sn = NeuralNetClassifier(
    module=SoupNetwork,
    module__layer_stock=layer_stock,
    module__soup_size=30,
    module__n_blocks=3,
    module__block_size=10,
    max_epochs=100,
    lr=0.1,
    iterator_train__shuffle=True,
    device='cuda',
    callbacks=callbacks
)


skorch_sn.fit(fmnist_train.train_data[:, None, :, :].float(), fmnist_train.train_labels.long())


