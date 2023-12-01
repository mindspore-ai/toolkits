import troubleshooter as ts
import time
import numpy as np
import torch
import pytest

from tempfile import TemporaryDirectory
from pathlib import Path
from troubleshooter.common.util import find_file


class NetWithSaveGrad(torch.nn.Module):
    def __init__(self, path):
        super(NetWithSaveGrad, self).__init__()
        self.dense = torch.nn.Linear(3, 2)
        self.apply(self._init_weights)
        self.path = path

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.zero_()
            module.bias.data.zero_()

    def forward(self, x):
        x = self.dense(x)
        x = ts.save_grad(self.path, x)
        return x


class NetWithParameterSaveGrad(torch.nn.Module):
    def __init__(self):
        super(NetWithParameterSaveGrad, self).__init__()
        self.dense = torch.nn.Linear(3, 2)
        self.apply(self._init_weights)
        self.dense_grad = torch.nn.Parameter(torch.zeros(
            (2), dtype=torch.float32), requires_grad=True)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.zero_()
            module.bias.data.zero_()

    def forward(self, x):
        x = self.dense(x)
        x = x + self.dense_grad
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_torch_save_grad_should_equal_to_parameter_save():
    temp_dir = TemporaryDirectory()
    path = Path(temp_dir.name)
    file_name = path / "dense"
    data = np.array([0.2, 0.5, 0.2], dtype=np.float32)
    label = np.array([1, 0], dtype=np.float32)
    label_pt = np.array([0], dtype=np.float32)

    net1 = NetWithSaveGrad(str(file_name))
    net2 = NetWithParameterSaveGrad()
    loss_fun = torch.nn.CrossEntropyLoss()

    out1 = net1(torch.tensor(data))
    out1 = torch.unsqueeze(out1, 0)
    loss2 = loss_fun(out1, torch.tensor(label_pt, dtype=torch.long))
    loss2.backward()

    out2 = net2(torch.tensor(data))
    out2 = torch.unsqueeze(out2, 0)
    loss2 = loss_fun(out2, torch.tensor(label_pt, dtype=torch.long))
    loss2.backward()
    time.sleep(0.1)
    file_list = find_file(path)
    assert np.allclose(
        np.load(path / file_list[0]), net2.dense_grad.grad.detach().cpu().numpy())
