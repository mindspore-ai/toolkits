import os
import shutil

import numpy as np

import troubleshooter as ts
import torch
import mindspore as ms
import tempfile
import pytest
from troubleshooter.migrator.save import _ts_save_cnt
from tests.util import check_delimited_list


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_compare_npy_dir(capsys):
    data1 = np.random.rand(1, 3, 2).astype(np.float32)
    data2 = np.random.rand(1, 3, 2).astype(np.float32)
    path1 = "/tmp/troubleshooter_ta/"
    path2 = "/tmp/troubleshooter_tb/"
    is_exists = os.path.exists(path1)
    if not is_exists:
        os.makedirs(path1)

    is_exists = os.path.exists(path2)
    if not is_exists:
        os.makedirs(path2)

    np.save('/tmp/troubleshooter_ta/data1.npy', data1)
    np.save('/tmp/troubleshooter_ta/data2.npy', data2)

    np.save('/tmp/troubleshooter_tb/data1.npy', data1)
    np.save('/tmp/troubleshooter_tb/data2.npy', data1)

    ts.migrator.compare_npy_dir(path1, path2)
    result = capsys.readouterr().out

    shutil.rmtree(path1)
    shutil.rmtree(path2)
    assert result.count('True') == 1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_compare_grads_part_match(capsys):
    _ts_save_cnt.reset()

    class PtSimpleNet(torch.nn.Module):
        def __init__(self):
            super(PtSimpleNet, self).__init__()
            self.fc = torch.nn.Linear(10, 5)
            self.bn = torch.nn.BatchNorm1d(5)

        def forward(self, x):
            return self.bn(self.fc(x))

    class MsSimpleNet(ms.nn.Cell):
        def __init__(self):
            super(MsSimpleNet, self).__init__()
            self.fc = ms.nn.Dense(10, 5)

        def construct(self, x):
            return self.fc(x)

    inputs = np.random.randn(32, 10).astype(np.float32)
    targets = np.random.randn(32, 5).astype(np.float32)
    pt_outpath = tempfile.mkdtemp(prefix="pt_")
    ms_outpath = tempfile.mkdtemp(prefix="ms_")
    pt_net = PtSimpleNet()
    pt_inputs = torch.tensor(inputs)
    pt_targets = torch.tensor(targets)
    pt_criterion = torch.nn.MSELoss()
    pt_optimizer = torch.optim.SGD(pt_net.parameters(), lr=0.01)
    pt_outputs = pt_net(pt_inputs)
    pt_loss = pt_criterion(pt_outputs, pt_targets)
    pt_optimizer.zero_grad()
    pt_loss.backward()
    pt_grads = ts.widget.get_pt_grads(pt_net)
    ts.save(os.path.join(pt_outpath, "torch_grads"), pt_grads)

    ms_net = MsSimpleNet()
    ms_inputs = ms.Tensor(inputs)
    ms_targets = ms.Tensor(targets)
    ms_loss_fn = ms.nn.MSELoss()

    def forward_fn(inputs, targets):
        out = ms_net(inputs)
        loss = ms_loss_fn(out, targets)
        return loss

    grad_fn = ms.value_and_grad(forward_fn, None, ms_net.trainable_params())
    ms_loss, ms_grads = grad_fn(ms_inputs, ms_targets)
    ts.save(os.path.join(ms_outpath, "ms_grads"), ms_grads)
    ts.migrator.compare_grads_dir(pt_outpath, ms_outpath)
    shutil.rmtree(pt_outpath)
    shutil.rmtree(ms_outpath)
    result = capsys.readouterr().out
    assert check_delimited_list(result, ['0_torch_grads_fc.weight_0.npy', '1_ms_grads_0.npy'])
    assert check_delimited_list(result, ['0_torch_grads_fc.bias_1.npy', '1_ms_grads_1.npy'])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_compare_grads_full_match(capsys):
    ms_path = '/tmp/ts_ms_test_grads/'
    pt_path = '/tmp/ts_pt_test_grads/'
    if not os.path.exists(ms_path):
        os.makedirs(ms_path)
    if not os.path.exists(pt_path):
        os.makedirs(pt_path)
    import troubleshooter as ts
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _ts_save_cnt.reset()

    class Net_PT(nn.Module):
        def __init__(self):
            super(Net_PT, self).__init__()
            self.fc1 = nn.Linear(2, 10)
            self.fc2 = nn.Linear(10, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            return x

    net_pt = Net_PT()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net_pt.parameters(), lr=0.1)

    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    labels = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    optimizer.zero_grad()
    outputs = net_pt(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    ts.save(pt_path + "grads.npy", ts.widget.get_pt_grads(net_pt))
    import mindspore.nn as nn
    from mindspore import Tensor
    import mindspore.common.dtype as mstype

    class Net_MS(nn.Cell):
        def __init__(self):
            super(Net_MS, self).__init__()
            self.fc1 = nn.Dense(2, 10)
            self.fc2 = nn.Dense(10, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def construct(self, x):
            x = self.relu(self.fc1(x))
            x = self.sigmoid(self.fc2(x))
            return x

    net_ms = Net_MS()
    loss_fn = nn.BCELoss()
    optimizer = nn.Momentum(net_ms.trainable_params(), learning_rate=0.1, momentum=0.9)

    inputs = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=mstype.float32)
    labels = Tensor([[0], [1], [1], [0]], dtype=mstype.float32)

    def forward_fn(data, label):
        logits = net_ms(data)
        loss = loss_fn(logits, label)
        return loss, logits

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_one_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        ts.save(ms_path + "grads.npy", grads)
        return loss

    train_one_step(inputs, labels)
    ts.migrator.compare_grads_dir(pt_path, ms_path)
    result = capsys.readouterr().out
    key_result = ['0_grads_fc2.bias_3.npy', '1_grads_3.npy']
    shutil.rmtree(ms_path)
    shutil.rmtree(pt_path)
    assert check_delimited_list(result, key_result)
