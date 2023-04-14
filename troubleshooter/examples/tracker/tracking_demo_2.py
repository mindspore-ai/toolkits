import os
import mindspore
import numpy as np
from mindspore import nn, ops, Tensor
import troubleshooter as ts

mindspore.set_context(mode=mindspore.PYNATIVE_MODE)

@ts.tracking(level=1, path_wl=['layer/conv.py'])
def main():
    conv = nn.Conv2d(3, 2, 3, pad_mode='valid', weight_init="ones")
    relu = nn.ReLU()
    seq = nn.SequentialCell([conv, relu])
    x = Tensor(np.ones([1, 3, 4, 4]), dtype=mindspore.float32)\

main()