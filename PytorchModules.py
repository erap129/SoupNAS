from functools import partial
from torch import nn


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size


class SoupConv:
    def __init__(self, input_channels):
        self.conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)
        self.out_channels




