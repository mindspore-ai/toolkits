#!/usr/bin/env python3
# coding=utf-8

"""
This is a python code template
"""

import os
from mindspore import Tensor, ops, nn
import numpy as np
import mindspore

import troubleshooter as ts


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.binary_cross_entropy = ops.BinaryCrossEntropy()

    def construct(self, logits, labels, weight):
        result = self.binary_cross_entropy(logits, labels, weight)
        return result


with ts.proposal(write_file_path="/tmp/"):
    logits = Tensor(np.random.uniform(0, 1, (4, 3, 388, 388)).astype(np.float32))
    labels = Tensor(np.random.randint(0, 2, (4, 2, 388, 388)).astype(np.float32))
    weight = Tensor(np.array([4, 3, 388, 388]), mindspore.float32)
    net = Net()
    out = net(logits, labels, weight)
    print('out:', out)
