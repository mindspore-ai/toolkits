import pytest
import mindspore.nn as m_nn
import torch.nn as t_nn
import torch
import mindspore as ms
import troubleshooter as ts
import numpy as np
import pytest


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

class ConstTorch(t_nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = t_nn.Linear(12, 21)
    
    def forward(self, x):
        return self.net1(x) + 1.

class ConstMS(m_nn.Cell):
    def __init__(self):
        super().__init__()
        self.net1 = m_nn.Dense(12, 21)
    
    def construct(self, x):
        return self.net1(x) + 1.4

@pytest.mark.skip
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_model(capsys):
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
    )
    diff_finder.compare()
    out, err = capsys.readouterr()
    key_result = 'test0-result_0 | test0-a |          True         |    100.00   |      1.00000'
    assert out.count(key_result) == 1

@pytest.mark.skip
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_th_tensor(capsys):
    input1 = (torch.rand([1, 12], dtype=torch.float32),
              torch.rand([1, 13], dtype=torch.float32))
    input2 = (torch.rand([1, 12], dtype=torch.float32),
              torch.randn([1, 13], dtype=torch.float32))
    ms_net = MSNet()
    pt_net = TorchNet()
    diff_finder = ts.NetDifferenceFinder(
        pt_net=pt_net,
        ms_net=ms_net,
        inputs=[input1, input2],
    )
    diff_finder.compare()
    out, err = capsys.readouterr()
    key_result = 'test0-result_0 | test0-a |          True         |    100.00   |      1.00000'
    assert out.count(key_result) == 1

@pytest.mark.skip
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_tensor(capsys):
    input1 = (ms.Tensor(np.random.randn(1, 12).astype(np.float32)), 
            ms.Tensor(np.random.randn(1, 13).astype(np.float32)))
    input2 = (ms.Tensor(np.random.randn(1, 12).astype(np.float32)), 
            ms.Tensor(np.random.randn(1, 13).astype(np.float32)))
    ms_net = MSNet()
    pt_net = TorchNet()
    diff_finder = ts.NetDifferenceFinder(
        pt_net=pt_net,
        ms_net=ms_net,
        inputs=[input1, input2],
    )
    diff_finder.compare()
    out, err = capsys.readouterr()
    key_result = 'test0-result_0 | test0-a |          True         |    100.00   |      1.00000'
    assert out.count(key_result) == 1

@pytest.mark.skip
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_single_input(capsys):
    ms_net = MSNet()
    pt_net = TorchNet()
    diff_finder = ts.NetDifferenceFinder(
        pt_net=pt_net,
        ms_net=ms_net,
        inputs=(np.ones([2, 12]).astype(np.float32), np.ones([2, 13]).astype(np.float32)),
        auto_conv_ckpt=False
    )
    diff_finder.compare()
    out, err = capsys.readouterr()
    print(out)
    key_result = "|     0.00    |      0.51057      | ['0.447508', '0.996762', '0.002890'] |"
    assert out.count(key_result)

@pytest.mark.skip
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_single_autoinput(capsys):
    pt_net = TorchNet()
    ms_net = MSNet()
    diff_finder = ts.NetDifferenceFinder(
        pt_net=pt_net,
        ms_net=ms_net,
        auto_input=(((1, 12), np.float32), ((1, 13), np.float32)))
    diff_finder.compare()
    out, err = capsys.readouterr()
    key_result = '| test0-result_0 | test0-a |          True         |    100.00   |'
    assert out.count(key_result)

@pytest.mark.skip
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_autoinput(capsys):
    pt_net = TorchNet()
    ms_net = MSNet()
    diff_finder = ts.NetDifferenceFinder(
        pt_net=pt_net,
        ms_net=ms_net,
        auto_input={'input': (((1, 12), np.float32), ((1, 13), np.float32)), 
                    'num': 2}
        )
    diff_finder.compare()
    out, err = capsys.readouterr()
    key_result = '| test0-result_0 | test0-a |          True         |    100.00   |      1.00000      |'
    assert out.count(key_result)

@pytest.mark.skip
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_diff(capsys):
    pt_net = ConstTorch()
    ms_net = ConstMS()
    diff_finder = ts.NetDifferenceFinder(
        pt_net=pt_net,
        ms_net=ms_net,
        auto_input=(((1, 12), np.float32), ))
    diff_finder.compare()
    out, err = capsys.readouterr()
    key_result = '| test0-result | test0-result |         False         |     0.00    |'
    assert out.count(key_result)
