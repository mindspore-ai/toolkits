import os
import tempfile
import shutil

import pytest
import troubleshooter as ts
import torch.nn as t_nn
import torch
import mindspore as ms
import mindspore.nn as m_nn
import numpy as np
from tests.util import check_delimited_list

ts.fix_random()

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
    diff_finder = ts.migrator.NetDifferenceFinder(
        pt_net=pt_net,
        ms_net=ms_net
    )
    diff_finder.compare(inputs=[input1, input2])
    out, err = capsys.readouterr()
    key_result = ['step_0_a', 'step_0_out_0', 'True', '100.00%', '1.00000']
    assert check_delimited_list(out, key_result)


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
    diff_finder = ts.migrator.NetDifferenceFinder(
        pt_net=pt_net,
        ms_net=ms_net
    )
    diff_finder.compare(inputs=[input1, input2])
    out, err = capsys.readouterr()
    key_result = ['step_0_a', 'step_0_out_0', 'True', '100.00%', '1.00000']
    assert check_delimited_list(out, key_result)


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
    diff_finder = ts.migrator.NetDifferenceFinder(
        pt_net=pt_net,
        ms_net=ms_net
    )
    diff_finder.compare(inputs=[input1, input2])
    out, err = capsys.readouterr()
    key_result = ['step_0_a', 'step_0_out_0', 'True', '100.00%', '1.00000']
    assert check_delimited_list(out, key_result)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_single_input(capsys):
    ms_net = MSNet()
    pt_net = TorchNet()
    diff_finder = ts.migrator.NetDifferenceFinder(
        pt_net=pt_net,
        ms_net=ms_net)
    diff_finder.compare(inputs=(np.ones([2, 12]).astype(np.float32), np.ones([2, 13]).astype(np.float32)))
    out, err = capsys.readouterr()
    key_result = ['True', '100.00%', '1.00000']
    assert check_delimited_list(out, key_result)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_single_autoinput(capsys):
    pt_net = TorchNet()
    ms_net = MSNet()
    diff_finder = ts.migrator.NetDifferenceFinder(
        pt_net=pt_net,
        ms_net=ms_net)
    diff_finder.compare(auto_inputs=(((1, 12), np.float32), ((1, 13), np.float32)))
    out, err = capsys.readouterr()
    key_result = ['step_0_a', 'step_0_out_0', 'True', '100.00']
    assert check_delimited_list(out, key_result)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_autoinput(capsys):
    pt_net = TorchNet()
    ms_net = MSNet()
    diff_finder = ts.migrator.NetDifferenceFinder(
        pt_net=pt_net,
        ms_net=ms_net)
    diff_finder.compare(auto_inputs={'input': (((1, 12), np.float32), ((1, 13), np.float32)), 
                    'num': 2})
    out, err = capsys.readouterr()
    key_result = ['step_0_a', 'step_0_out_0', 'True', '100.00%', '1.00000']
    assert check_delimited_list(out, key_result)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_diff(capsys):
    pt_net = ConstTorch()
    ms_net = ConstMS()
    diff_finder = ts.migrator.NetDifferenceFinder(
        pt_net=pt_net,
        ms_net=ms_net)
    diff_finder.compare(auto_inputs=(((1, 12), np.float32), ))
    out, err = capsys.readouterr()
    key_result = ['step_0_out', 'step_0_out', 'False', '0.00']
    assert check_delimited_list(out, key_result)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_load_weight_file(capsys):
    tmp = tempfile.mkdtemp(prefix="net_diff_load_weight")
    pt_net = ConstTorch()
    ms_net = ConstMS()
    torch.save(pt_net.state_dict(), os.path.join(tmp, 'torch.pth'))
    ms.save_checkpoint(ms_net, os.path.join(tmp, 'mindpsore.ckpt'))
    diff_finder = ts.migrator.NetDifferenceFinder(
        pt_net=pt_net,
        ms_net=ms_net,
        pt_params_path=os.path.join(tmp, 'torch.pth'),
        ms_params_path=os.path.join(tmp, 'mindpsore.ckpt'))
    diff_finder.compare(auto_inputs=(((1, 12), np.float32), ))
    out, err = capsys.readouterr()
    key_result = ['step_0_out', 'step_0_out', 'False', '0.00']
    shutil.rmtree(tmp)
    assert check_delimited_list(out, key_result)
