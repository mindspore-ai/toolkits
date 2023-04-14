#!/usr/bin/env python3

import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import context

import troubleshooter as ts
from mindspore import Tensor
import numpy as np

context.set_context(device_target='CPU')
class Net(nn.Cell):
    def __init__(self, axis, keep_dims):
        super().__init__()
        self.reducesum = ops.ReduceSum(keep_dims=keep_dims)
        self.axis = axis

    def construct(self, input_x):
        return self.reducesum(input_x, self.axis)


@ts.proposal(write_file_path="/tmp/")
def main():
    x = Tensor(np.random.randn(10, 5, 4, 4, 4, 4, 4, 4, 4, 4), mindspore.float32)
    net = Net(axis=(1,), keep_dims=True)
    out = net(x)
    print("out", out.shape)


if __name__ == "__main__":
    main()
