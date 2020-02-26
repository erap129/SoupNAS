from functools import partial
from torch import nn
from torch.nn import Conv2d
import torch


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size


class ResidualBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResidualBlock, self).__init__()

        # TODO: 3x3 convolution -> relu
        # the input and output channel number is channel_num
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, padding=1),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, padding=1),
            nn.BatchNorm2d(channel_num),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # TODO: forward
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual
        out = self.relu(x)
        return out


class ConcatThenOneByOne(nn.Module):
    def __init__(self, channel_num, num_inputs):
        super(ConcatThenOneByOne, self).__init__()
        self.onebyone = Conv2d(channel_num * num_inputs, channel_num, kernel_size=1)

    def forward(self, *input):
        return self.onebyone(torch.cat(input, 1))


class SoupConv:
    def __init__(self, input_channels):
        self.conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)
        self.out_channels




