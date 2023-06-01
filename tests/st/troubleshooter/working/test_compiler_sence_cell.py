from mindspore import nn, context
from mindspore import ops, set_context
import mindspore as ms

import troubleshooter as ts

#context.set_context(mode=ms.GRAPH_MODE, pynative_synchronize=False)

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def construct(self, x):
        return self.flatten(x)


@ts.proposal(write_file_path="/tmp/")
def test_compile_cell():
    net = Net()
    print("network:")
    print(net(2))


if __name__ == "__main__":
    test_compile_cell()
