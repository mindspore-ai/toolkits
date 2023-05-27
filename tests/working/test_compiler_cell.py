from mindspore import nn, Tensor
from mindspore import ops, set_context
import mindspore as ms

import troubleshooter as ts

set_context(mode=ms.PYNATIVE_MODE)


# can't catch python code compile errors.
# @ts.proposal(write_file_path="/tmp/")
class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def construct(self, x):
        return self.flatten(x)


@ts.proposal(write_file_path="/tmp/")
def test_compile_cell():
    net = Net()
    input1 = Tensor(3, ms.float32)
    print(net(input1))


if __name__ == "__main__":
    test_compile_cell()
