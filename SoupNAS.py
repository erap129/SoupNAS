import ast
import os
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
from PytorchModules import Conv2dAuto, ResidualBlock, ConcatThenOneByOne
from graphviz import Digraph
from datetime import datetime
from sacred import Experiment
import pandas as pd
import sys


ex = Experiment()
if 'debug' not in sys.argv:
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


class OptionsSoupNetwork(nn.Module):
    def __init__(self, layer_stock, soup_size, n_blocks, block_size, layer_options):
        super(OptionsSoupNetwork, self).__init__()
        self.layer_stock = layer_stock
        self.soup_size = soup_size
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.expansion_convs = ModuleList([Conv2dAuto(in_channels=max(1, 32 * i), out_channels=32 * (i + 1), kernel_size=3) for i in range(n_blocks)])
        self.layer_options = layer_options
        self.soup = self.make_options_soup()
        self.linear = nn.Linear(int(x_train.shape[2] / 2 ** n_blocks) * int(x_train.shape[3] / 2 ** n_blocks) * int(32 * (n_blocks)), 10)

    def make_options_soup(self):
        soup = ModuleList()
        for block_idx in range(self.n_blocks):
            block_channels = max(1, 32*block_idx)
            soup.append(ModuleList())
            for i in range(self.block_size):
                soup[-1].append(ModuleList())
                for opt_idx in range(self.layer_options):
                    random_layer = random.sample(layer_stock, 1)[0]
                    soup[-1][-1].append(get_soup_layer(random_layer, block_channels, block_channels))
        return soup

    def forward(self, X):
        for block_idx, block in enumerate(self.soup):
            for options in block:
                layer = random.choice(options)
                X = layer(X)
            X = self.expansion_convs[block_idx](X)
            X = nn.MaxPool2d(2, stride=2)(X)
        X = X.view(X.size(0), -1)
        X = self.linear(X)
        X = nn.functional.softmax(X)
        return X


class LinearSoupNetwork(nn.Module):
    def __init__(self, layer_stock, soup_size, n_blocks, block_size, random_input_factor, num_receptors, paths):
        super(LinearSoupNetwork, self).__init__()
        self.layer_stock = layer_stock
        self.soup_size = soup_size
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.num_receptors = num_receptors
        self.random_input_factor = random_input_factor
        self.expansion_convs = ModuleList([Conv2dAuto(in_channels=max(1, 32 * i), out_channels=32 * (i + 1), kernel_size=3) for i in range(n_blocks)])
        self.paths = paths
        self.soup = self.make_linear_soup()
        self.linear = nn.Linear(int(x_train.shape[2] / 2 ** n_blocks) * int(x_train.shape[3] / 2 ** n_blocks) * int(32 * (n_blocks)), 10)

    def plot_soup(self, filepath):
        f = Digraph('SoupNAS model')
        for block_idx in range(self.n_blocks):
            for key in self.soup[str(block_idx)]:
                if 'connector' not in key:
                    node_name = f'{block_idx}_{key.split("_")[0]}'
                    f.node(node_name)
                    for incoming in intlst_to_strlst(ast.literal_eval(key.split('_')[1])):
                        f.edge(f'{block_idx}_{incoming}', node_name)
        f.render(filepath, view=False)

    def make_linear_soup(self):
        soup = ModuleDict()
        for block_idx in range(self.n_blocks):
            block_channels = max(1, 32*block_idx)
            soup[str(block_idx)] = ModuleDict()
            for i in range(self.block_size):
                if i > 0:
                    prev_layers = list(range(max(-1, i-self.random_input_factor[block_idx][i]), i))
                else:
                    prev_layers = [-1]
                random_layer = random.sample(layer_stock, 1)[0]
                layer = get_soup_layer(random_layer, block_channels, block_channels)
                layer.num_receptors = self.num_receptors[block_idx][i]
                soup[str(block_idx)][f'{i}_{prev_layers}'] = layer
                if layer.num_receptors > 1:
                    soup[str(block_idx)][f'{i}_{prev_layers}_connector'] = \
                        ConcatThenOneByOne(block_channels, layer.num_receptors)
        block_channels += 32
        if self.paths > 1:
            soup['final_connector'] = ConcatThenOneByOne(block_channels, self.paths)
        return soup

    def forward(self, X):
        if self.paths == 1:
            return self.forward_regular(X)
        else:
            return self.forward_paths(X)

    def forward_regular(self, X):
        run_str = 'input->'
        for block_idx in range(self.n_blocks):
            block_outputs = {}
            X.tag = '-1'
            run_str += f'BLOCK_{block_idx}['
            for step in range(self.block_size):
                relevant_layers = {k: v for k, v in self.soup[str(block_idx)].items() if
                                   'connector' not in k and X.tag in intlst_to_strlst(
                                       ast.literal_eval(k.split('_')[1]))}
                if len(relevant_layers) > 0:
                    layer_key = random.choice(list(relevant_layers.keys()))
                    layer = self.soup[str(block_idx)][layer_key]
                    if layer.num_receptors == 1:
                        X = layer(X)
                    else:
                        relevant_inputs = [v for k, v in block_outputs.items() if k != X.tag and k in intlst_to_strlst(
                            ast.literal_eval(layer_key.split('_')[1]))]
                        if len(relevant_inputs) >= layer.num_receptors - 1:
                            inputs = [X] + random.sample([v for k, v in block_outputs.items() if k != X.tag and
                                                          k in intlst_to_strlst(
                                ast.literal_eval(layer_key.split('_')[1]))],
                                                         layer.num_receptors - 1)
                            X = self.soup[str(block_idx)][f'{layer_key}_connector'](*inputs)
                        else:
                            X = layer(X)
                    X.tag = layer_key.split('_')[0]
                    block_outputs[X.tag] = X
                    run_str += f'{X.tag}->'
            run_str += ']->'
            save_tag = X.tag
            X = self.expansion_convs[block_idx](X)
            X = nn.MaxPool2d(2, stride=2)(X)
            X.tag = save_tag
        X = X.view(X.size(0), -1)
        X = self.linear(X)
        X = nn.functional.softmax(X)
        return X

    def forward_paths(self, X):
        for block_idx in range(self.n_blocks):
            X.tag = '-1'
            block_outputs = [{X.tag: X} for i in range(self.paths)]
            for step in range(self.block_size * self.paths):
                relevant_layers = {}
                for i in range(len(block_outputs)):
                    relevant_layers[i] = {k: v for k, v in self.soup[str(block_idx)].items() if
                                        'connector' not in k and list(block_outputs[i].keys())[0] in intlst_to_strlst(ast.literal_eval(k.split('_')[1]))}
                relevant_layers = {k: v for k, v in relevant_layers.items() if len(v) > 0}
                if len(relevant_layers) > 0:
                    chosen_path = random.choice(list(relevant_layers.keys()))
                    layer_key = random.choice(list(relevant_layers[chosen_path].keys()))
                    layer = self.soup[str(block_idx)][layer_key]
                    if layer.num_receptors == 1:
                        X = layer(list(block_outputs[chosen_path].values())[0])
                    else:
                        relevant_inputs = []
                        for i in range(len(block_outputs)):
                            relevant_inputs.extend([v for k, v in block_outputs[i].items() if k != X.tag and k in intlst_to_strlst(ast.literal_eval(layer_key.split('_')[1]))])
                        if len(relevant_inputs) >= layer.num_receptors - 1:
                            inputs = [X] + random.sample(relevant_inputs, layer.num_receptors - 1)
                            X = self.soup[str(block_idx)][f'{layer_key}_connector'](*inputs)
                        else:
                            X = layer(X)
                    X.tag = layer_key.split('_')[0]
                    block_outputs[chosen_path] = {X.tag: X}
            for i in range(len(block_outputs)):
                X = list(block_outputs[i].values())[0]
                save_tag = X.tag
                X = self.expansion_convs[block_idx](X)
                X = nn.MaxPool2d(2, stride=2)(X)
                X.tag = save_tag
                block_outputs[i] = {X.tag: X}
        X = self.soup['final_connector'](*[list(b.values())[0] for b in block_outputs])
        X = X.view(X.size(0), -1)
        X = self.linear(X)
        X = nn.functional.softmax(X)
        return X


def linear_experiment(callbacks, today_str):
    dict_list = []
    for random_input_factor in range(1, 10):
        for iteration in range(100):
            skorch_sn = NeuralNetClassifier(
                module=LinearSoupNetwork,
                module__layer_stock=layer_stock,
                module__soup_size=30,
                module__n_blocks=3,
                module__block_size=10,
                module__random_input_factor=[[random_input_factor for i in range(10)] for j in range(3)],
                module__num_receptors=[[1 for i in range(10)] for j in range(3)],
                max_epochs=50,
                lr=0.1,
                iterator_train__shuffle=True,
                device='cuda',
                callbacks=callbacks
            )
            skorch_sn.fit(fmnist_train.train_data[:, None, :, :].float(), fmnist_train.train_labels.long())
            y_pred = skorch_sn.predict(fmnist_test.test_data[:, None, :, :].float())
            accuracy = metrics.accuracy_score(fmnist_test.test_labels.long(), y_pred)
            dict_list.append({'random_input_factor': random_input_factor,
                              'iteration': iteration,
                              'accuracy': accuracy})
            pd.DataFrame(dict_list).to_csv(f'results/{today_str}.csv')
        ex.add_artifact(f'soup_{random_input_factor}.pdf')


def skip_experiment(callbacks, today_str):
    dict_list = []
    for two_receptor_layer in range(1, 10):
        for iteration in range(100):
            skorch_sn = NeuralNetClassifier(
                module=LinearSoupNetwork,
                module__layer_stock=layer_stock,
                module__soup_size=30,
                module__n_blocks=3,
                module__block_size=10,
                module__random_input_factor=[[1 for i in range(two_receptor_layer)] + [2] + [1 for i in range(two_receptor_layer+1, 10)] for j in range(3)],
                module__num_receptors=[[1 for i in range(two_receptor_layer)] + [2] + [1 for i in range(two_receptor_layer+1, 10)] for j in range(3)],
                max_epochs=50,
                lr=0.1,
                iterator_train__shuffle=True,
                device='cuda',
                callbacks=callbacks
            )
            skorch_sn.fit(fmnist_train.train_data[:, None, :, :].float(), fmnist_train.train_labels.long())
            y_pred = skorch_sn.predict(fmnist_test.test_data[:, None, :, :].float())
            accuracy = metrics.accuracy_score(fmnist_test.test_labels.long(), y_pred)
            dict_list.append({'two_receptor_layer': two_receptor_layer,
                              'iteration': iteration,
                              'accuracy': accuracy})
            pd.DataFrame(dict_list).to_csv(f'results/{today_str}/skip_experiment.csv')
        plot_filename = f'results/{today_str}/soup_{two_receptor_layer}'
        try:
            skorch_sn.module_.plot_soup(plot_filename)
        except Exception as e:
            print('pydot doesnt exist on this system, not plotting')
        ex.add_artifact(plot_filename)


def skip_linear_experiment(callbacks, today_str):
    dict_list = []
    for random_input_factor in range(1, 10):
        for two_receptor_layer in range(1, 10):
            for iteration in range(100):
                skorch_sn = NeuralNetClassifier(
                    module=LinearSoupNetwork,
                    module__layer_stock=layer_stock,
                    module__soup_size=30,
                    module__n_blocks=3,
                    module__block_size=10,
                    module__random_input_factor=[[random_input_factor for i in range(two_receptor_layer)] + [max(2, random_input_factor)] + [random_input_factor for i in range(two_receptor_layer+1, 10)] for j in range(3)],
                    module__num_receptors=[[1 for i in range(two_receptor_layer)] + [2] + [1 for i in range(two_receptor_layer+1, 10)] for j in range(3)],
                    max_epochs=50,
                    lr=0.1,
                    iterator_train__shuffle=True,
                    device='cuda',
                    callbacks=callbacks
                )
                skorch_sn.fit(fmnist_train.train_data[:, None, :, :].float(), fmnist_train.train_labels.long())
                y_pred = skorch_sn.predict(fmnist_test.test_data[:, None, :, :].float())
                accuracy = metrics.accuracy_score(fmnist_test.test_labels.long(), y_pred)
                dict_list.append({'random_input_factor': random_input_factor,
                                    'two_receptor_layer': two_receptor_layer,
                                  'iteration': iteration,
                                  'accuracy': accuracy})
                pd.DataFrame(dict_list).to_csv(f'results/{today_str}/skip_linear_experiment.csv')


def paths_experiment(callbacks, today_str):
    dict_list = []
    for paths in range(1, 10):
        for iteration in range(100):
            skorch_sn = NeuralNetClassifier(
                module=LinearSoupNetwork,
                module__layer_stock=layer_stock,
                module__soup_size=30,
                module__n_blocks=3,
                module__block_size=10,
                module__random_input_factor=[[1 for i in range(10)] for j in range(3)],
                module__num_receptors=[[1 for i in range(10)] for j in range(3)],
                module__paths=paths,
                max_epochs=1,
                lr=0.1,
                iterator_train__shuffle=True,
                device='cuda',
                callbacks=callbacks
            )
            skorch_sn.fit(fmnist_train.train_data[:, None, :, :].float(), fmnist_train.train_labels.long())
            y_pred = skorch_sn.predict(fmnist_test.test_data[:, None, :, :].float())
            accuracy = metrics.accuracy_score(fmnist_test.test_labels.long(), y_pred)
            dict_list.append({'paths': paths,
                              'iteration': iteration,
                              'accuracy': accuracy})
            pd.DataFrame(dict_list).to_csv(f'results/{today_str}/paths_experiment.csv')


def options_experiment(callbacks, today_str):
    dict_list = []
    for layer_options in range(1, 10):
            for iteration in range(1):
                skorch_sn = NeuralNetClassifier(
                    module=OptionsSoupNetwork,
                    module__layer_stock=layer_stock,
                    module__soup_size=30,
                    module__n_blocks=3,
                    module__block_size=10,
                    module__layer_options=layer_options,
                    max_epochs=1,
                    lr=0.1,
                    iterator_train__shuffle=True,
                    device='cuda',
                    callbacks=callbacks
                )
                skorch_sn.fit(fmnist_train.train_data[:, None, :, :].float(), fmnist_train.train_labels.long())
                y_pred = skorch_sn.predict(fmnist_test.test_data[:, None, :, :].float())
                accuracy = metrics.accuracy_score(fmnist_test.test_labels.long(), y_pred)
                dict_list.append({'layer_options': layer_options,
                                  'iteration': iteration,
                                  'accuracy': accuracy})
                pd.DataFrame(dict_list).to_csv(f'results/{today_str}.csv')


@ex.main
def my_main():
    callbacks = []
    writer = SummaryWriter()
    callbacks.append(MyTensorBoard(writer, X=fmnist_train.train_data[:, None, :, :].float()[:2]))
    callbacks.append(ShuffleOrder())
    callbacks.append(EarlyStopping())
    exp = sys.argv[1]
    today = datetime.now()
    today_str = today.strftime("%d-%m-%H-%M")
    os.mkdir(f'results/{today_str}')
    dict_list = []
    if exp == 'linear':
        linear_experiment(callbacks, today_str)
    elif exp == 'options':
        options_experiment(callbacks, today_str)
    elif exp == 'skip':
        skip_experiment(callbacks, today_str)
    elif exp == 'skip_linear':
        skip_linear_experiment(callbacks, today_str)
    elif exp == 'paths':
        paths_experiment(callbacks, today_str)
    filename = 'SoupNAS_fashion_mnist.csv'
    pd.DataFrame(dict_list).to_csv(filename)
    ex.add_artifact(filename)


if __name__ == '__main__':
    ex.run()


