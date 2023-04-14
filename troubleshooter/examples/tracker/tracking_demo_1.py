import os
import mindspore
import numpy as np
from mindspore import nn, ops, Tensor
import troubleshooter as ts

mindspore.set_context(mode=mindspore.PYNATIVE_MODE)


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.sqrt = ops.Sqrt()
        self.matmul = ops.MatMul()

    def construct(self, input_x):
        y = self.matmul(input_x, input_x)
        x = self.sqrt(y)
        return x


@ts.tracking(level=2, check_mode=2, check_keyword='nan')
def nan_func():
    input_x = Tensor(np.array([[0.0, -1.0], [4.0, 3.0]]))
    k = 3.0
    net = Net()
    print(net(input_x))


nan_func()
