import shutil
import os

import pytest
import troubleshooter as ts
import torch.nn as t_nn
import mindspore.nn as m_nn

class ConstTorch(t_nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = t_nn.Linear(12, 21)
    
    def forward(self, x):
        return self.net1(x)

class ConstMS(m_nn.Cell):
    def __init__(self):
        super().__init__()
        self.net1 = m_nn.Dense(12, 21)
    
    def construct(self, x):
        return self.net1(x)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_save_net_and_weight_params():
    pt_net = ConstTorch()
    ms_net = ConstMS()
    ts.widget.save_net_and_weight_params(pt_net, path="pt")
    ts.widget.save_net_and_weight_params(ms_net, path="ms")
    pt = ["torch_troubleshooter_create.pth", "torch_net_architecture.txt", "torch_net_map.json"]
    ms = ["mindspore_troubleshooter_create.ckpt", "mindspore_net_architecture.txt"]
    for file in pt:
        assert os.path.isfile(os.path.join("pt", file))
    for file in ms:
        assert os.path.isfile(os.path.join("ms", file))
    shutil.rmtree("pt")
    shutil.rmtree("ms")
