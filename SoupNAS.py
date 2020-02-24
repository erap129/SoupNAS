import ast
import random

import torchvision.datasets as datasets
from sacred.observers import MongoObserver
from sklearn import metrics
from torch.nn import BatchNorm2d, Dropout, ModuleDict, ModuleList
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
from skorch.callbacks import TensorBoard, Callback, EarlyStopping
from torch import nn
from skorch import NeuralNetClassifier
from PytorchModules import Conv2dAuto, ResidualBlock
from graphviz import Digraph
from datetime import date
from sacred import Experiment
import pandas as pd


ex = Experiment()
ex.observers.append(MongoObserver.create(url=f'mongodb://132.72.81.248/SoupNAS', db_name='SoupNAS'))
writer = SummaryWriter()
fmnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(fmnist_train, batch_size=4, shuffle=True)
testloader = torch.utils.data.DataLoader(fmnist_test, batch_size=4, shuffle=False)
x_train = fmnist_train.train_data[:, None, :, :].float()
y_train = fmnist_train.train_labels.long()

layer_stock = {ResidualBlock, Conv2dAuto, BatchNorm2d, nn.ReLU, Dropout}


def intlst_to_strlst(int_list):
    return [str(item) for item in int_list]


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
        # self.add_graph(net.module_, self.X)
        # print(net.module_.current_layers)
        pass

class ShuffleOrder(Callback):
    def on_epoch_end(self, net, **kwargs):
        # net.module_.shuffle_net()
        pass

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
    def __init__(self, layer_stock, soup_size, n_blocks, block_size, random_input_range, random_input_factor):
        super(SoupNetwork, self).__init__()
        self.layer_stock = layer_stock
        self.soup_size = soup_size
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.random_input_range = random_input_range
        self.random_input_factor = random_input_factor
        self.expansion_convs = ModuleList([Conv2dAuto(in_channels=max(1, 32 * i), out_channels=32 * (i + 1), kernel_size=3) for i in range(n_blocks)])
        self.soup = self.make_linear_soup()
        self.plot_soup()
        # self.shuffle_net()
        self.linear = nn.Linear(int(x_train.shape[2] / 2 ** n_blocks) * int(x_train.shape[3] / 2 ** n_blocks) * int(32 * (n_blocks)), 10)

    def plot_soup(self):
        f = Digraph('SoupNAS model')
        for block_idx in range(self.n_blocks):
            for key in self.soup[str(block_idx)]:
                node_name = f'{block_idx}_{key.split("_")[0]}'
                f.node(node_name)
                for incoming in intlst_to_strlst(ast.literal_eval(key.split('_')[1])):
                    f.edge(f'{block_idx}_{incoming}', node_name)
        f.render(f'soup_{self.random_input_factor}_{self.random_input_range}', view=False)

    def make_linear_soup(self):
        soup = ModuleDict()
        for block_idx in range(self.n_blocks):
            block_channels = max(1, 32*block_idx)
            soup[str(block_idx)] = ModuleDict()
            for i in range(self.block_size):
                if i > 0:
                    # prev_layers = random.sample(list(range(max(0, i-self.random_input_range), i)), min(i, random.randint(1, self.random_input_factor)))
                    prev_layers = list(range(max(0, i-self.random_input_factor), i))
                else:
                    prev_layers = ['startblock']
                random_layer = random.sample(layer_stock, 1)[0]
                soup[str(block_idx)][f'{i}_{prev_layers}'] = get_soup_layer(random_layer, block_channels, block_channels)
        return soup

    def shuffle_net(self):
        self.current_layers = ModuleDict()
        for block_idx in range(self.n_blocks):
            self.current_layers[str(block_idx)] = ModuleList()
            for step in range(self.block_size):
                block_soup = self.soup[str(block_idx)]
                relevant_layers = [block_soup[k] for k in block_soup.keys() if
                                   step in ast.literal_eval(k.split('_')[2])]
                self.current_layers[str(block_idx)].append(random.choice(relevant_layers))

    def forward(self, X):
        run_str = 'input->'
        for block_idx in range(self.n_blocks):
            X.tag = 'startblock'
            run_str += f'BLOCK_{block_idx}['
            for step in range(self.block_size):
                relevant_layers = {k: v for k, v in self.soup[str(block_idx)].items() if X.tag in intlst_to_strlst(ast.literal_eval(k.split('_')[1]))}
                if len(relevant_layers) > 0:
                    layer_key = random.choice(list(relevant_layers.keys()))
                    layer = self.soup[str(block_idx)][layer_key]
                    X = layer(X)
                    X.tag = layer_key.split('_')[0]
                    run_str += f'{X.tag}->'
            run_str += ']->'
            save_tag = X.tag
            X = self.expansion_convs[block_idx](X)
            X = nn.MaxPool2d(2, stride=2)(X)
            X.tag = save_tag
        # print(run_str)
        X = X.view(X.size(0), -1)
        X = self.linear(X)
        X = nn.functional.softmax(X)
        return X


@ex.automain
def my_main():
    callbacks = []
    writer = SummaryWriter()
    callbacks.append(MyTensorBoard(writer, X=fmnist_train.train_data[:, None, :, :].float()[:2]))
    callbacks.append(ShuffleOrder())
    callbacks.append(EarlyStopping())
    today = date.today()
    today_str = today.strftime("%d-%m-%h-%m")
    dict_list = []
    for random_input_range in range(1, 2):
        for random_input_factor in range(1, 10):
            for iteration in range(100):
                skorch_sn = NeuralNetClassifier(
                    module=SoupNetwork,
                    module__layer_stock=layer_stock,
                    module__soup_size=30,
                    module__n_blocks=3,
                    module__block_size=10,
                    module__random_input_range=random_input_range,
                    module__random_input_factor=random_input_factor,
                    max_epochs=50,
                    lr=0.1,
                    iterator_train__shuffle=True,
                    device='cuda',
                    callbacks=callbacks
                )
                skorch_sn.fit(fmnist_train.train_data[:, None, :, :].float(), fmnist_train.train_labels.long())
                y_pred = skorch_sn.predict(fmnist_test.test_data[:, None, :, :].float())
                accuracy = metrics.accuracy_score(fmnist_test.test_labels.long(), y_pred)
                dict_list.append({'random_input_range': random_input_range,
                                  'random_input_factor': random_input_factor,
                                  'iteration': iteration,
                                  'accuracy': accuracy})
                pd.DataFrame(dict_list).to_csv(f'results/{today_str}.csv')
            ex.add_artifact(f'soup_{random_input_factor}_{random_input_range}.pdf')
    filename = 'SoupNAS_fashion_mnist.csv'
    pd.DataFrame(dict_list).to_csv(filename)
    ex.add_artifact(filename)


