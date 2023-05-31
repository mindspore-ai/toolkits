import numpy as np
import torch
import torch.nn as nn
from ptdbg_ascend import *
# from hooks import acc_cmp_dump, set_dump_switch
# from initialize import register_hook
from torch import Tensor


def conv(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    pad_mode="valid",
    has_bias=True,
):
    if pad_mode == "same" and stride == 1:
        padding = kernel_size // 2
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, padding=padding, bias=has_bias
    )


def fc_with_initialize(input_channels, out_channels, has_bias=True):
    return nn.Linear(input_channels, out_channels, bias=has_bias)


class DataNormTranspose(nn.Module):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respectively.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respectively.
    """

    def __init__(self, dataset_name="imagenet"):
        super(DataNormTranspose, self).__init__()
        # Computed from random subset of ImageNet training images
        if dataset_name == "imagenet":
            self.mean = Tensor(
                np.array([0.485 * 255, 0.456 * 255, 0.406 * 255]).reshape((1, 1, 1, 3))
            )
            self.std = Tensor(
                np.array([0.229 * 255, 0.224 * 255, 0.225 * 255]).reshape((1, 1, 1, 3))
            )
        else:
            self.mean = Tensor(np.array([0.4914, 0.4822, 0.4465]).reshape((1, 1, 1, 3)))
            self.std = Tensor(np.array([0.2023, 0.1994, 0.2010]).reshape((1, 1, 1, 3)))

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = x.permute(0, 3, 1, 2)
        return x


class AlexNet(nn.Module):
    """
    Alexnet
    """

    def __init__(
        self,
        num_classes=10,
        channel=3,
        phase="train",
        include_top=True,
        dataset_name="imagenet",
    ):
        super(AlexNet, self).__init__()
        self.data_trans = DataNormTranspose(dataset_name=dataset_name)
        self.conv1 = conv(channel, 64, 11, stride=4, has_bias=True)
        self.conv2 = conv(64, 128, 5, pad_mode="same", has_bias=True)
        self.conv3 = conv(128, 192, 3, pad_mode="same", has_bias=True)
        self.conv4 = conv(192, 256, 3, pad_mode="same", has_bias=True)
        self.conv5 = conv(256, 256, 3, pad_mode="same", has_bias=True)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2)
        self.include_top = include_top
        if self.include_top:
            dropout_ratio = 0.65
            if phase == "test":
                dropout_ratio = 1.0
            self.flatten = nn.Flatten()
            self.fc1 = fc_with_initialize(6 * 6 * 256, 4096)
            self.fc2 = fc_with_initialize(4096, 4096)
            self.fc3 = fc_with_initialize(4096, num_classes)
            self.dropout = nn.Dropout(p=1 - dropout_ratio)

    def forward(self, x):
        """define network"""
        x = self.data_trans(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # set_dump_switch("ON", mode="stack")
    module_generator = ModuleGenerator()
    module_generator.enable_ms()
    net = AlexNet()

    register_hook(net, acc_cmp_dump)
    set_dump_switch("ON")
    # grad_net = ms.grad(net)
    output = net(Tensor(np.random.random([1, 227, 227, 3]).astype(np.float32)))
    z=torch.sum(output)
    z.backward()
    print(output)
