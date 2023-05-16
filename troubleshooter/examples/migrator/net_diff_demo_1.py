import mindspore.nn as m_nn
import torch.nn as t_nn
import troubleshooter as ts
import numpy as np
import sys


class TorchNet(t_nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = t_nn.Linear(12, 21)
        self.net2 = t_nn.Linear(13, 22)

    def forward(self, x, y):
        return {'a': self.net1(x), 'b': self.net2(y)}


class MSNet(m_nn.Cell):
    def __init__(self, n=0) -> None:
        super().__init__()
        self.net1 = m_nn.Dense(12, 21)
        self.net2 = m_nn.Dense(13, 22)
        self.n = n

    def construct(self, x, y):
        if self.n:
            print(self.n)
        return self.net1(x), self.net2(y)


def main():
    input1 = (np.random.randn(1, 12).astype(np.float32),
              np.random.randn(1, 13).astype(np.float32))
    input2 = (np.random.randn(1, 12).astype(np.float32),
              np.random.randn(1, 13).astype(np.float32))
    ms_net = MSNet()
    pt_net = TorchNet()
    diff_finder = ts.NetDifferenceFinder(
        pt_net=pt_net,
        ms_net=ms_net,
        inputs=[input1, input2],
        #out_path='troubleshooter/tests/diff_handler/results',
        #print_result=False,
    )
    diff_finder.compare()

main()