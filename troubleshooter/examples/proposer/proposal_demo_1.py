import numpy as np
import mindspore
from mindspore import ops, Tensor, nn
import troubleshooter as ts

mindspore.set_context(mode=mindspore.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = ops.ReLU()
        self.reducesum = ops.ReduceSum()

    def construct(self, x, a, b):
        if a > b:
            return self.relu(x)  # shape: (2, 3, 4, 5), dtype:Float32
        else:
            return self.reducesum(x)  # shape:(), dype: Float32


with ts.proposal():
    input_x = Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
    input_a = Tensor(2, mindspore.float32)
    input_b = Tensor(6, mindspore.float32)
    net = Net()
    out = net(input_x, input_a, input_b)
    print(out)