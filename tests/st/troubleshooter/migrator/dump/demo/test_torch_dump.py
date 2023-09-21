import shutil
import tempfile
from pathlib import Path

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import troubleshooter as ts
from troubleshooter.migrator import api_dump_init, api_dump_start, api_dump_stop
from tests.st.troubleshooter.migrator.dump.utils import get_pkl_npy_stack_list


def generate_data():
    data_path = Path(tempfile.mkdtemp(prefix="test_data"))
    np.save(data_path / 'label.npy',
            np.random.randn(1, 10).astype(np.float32))
    np.save(data_path / 'data.npy',
            np.random.randn(1, 3, 90, 300).astype(np.float32))
    return data_path


class BaseTrainOneStep:
    def __init__(self, net, data_path, dump_path, info_path=None, retain_backward=True) -> None:
        self.net = net
        self.net.train()
        if info_path:
            ts.migrator.save_net_and_weight_params(self.net, path=info_path)
        self.data = torch.tensor(np.load(data_path/'data.npy'))
        self.label = torch.tensor(np.load(data_path/'label.npy'))
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01)
        self.optimizer.zero_grad()
        api_dump_init(self.net, dump_path, retain_backward=retain_backward)

    def __call__(self):
        api_dump_start()
        out = self.net(self.data)
        loss = self.criterion(out, self.label)
        loss.backward()
        self.optimizer.step()
        api_dump_stop()


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv = nn.Conv2d(3, 5, kernel_size=3, stride=3, padding=0)
        self.bn = nn.BatchNorm2d(5)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(15000, 10)


def train_pt_one_step_all(data_path, dump_path, info_path=None, retain_backward=True,
                          **api_dump_start_args):
    class Net(BaseNet):
        def forward(self, x):
            api_dump_start(**api_dump_start_args)
            x = self.conv(x)
            x = torch.clip(x, 0.2, 0.5)
            x = self.bn(x)
            x = self.relu(x)
            x = x.reshape(1, -1)
            x = self.linear(x)
            x = self.relu(x)
            return x
    train_one_step = BaseTrainOneStep(
        Net(), data_path, dump_path, info_path, retain_backward)
    train_one_step()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_api_dump_torch_all():
    data_path = generate_data()
    dump_path = Path(tempfile.mkdtemp(prefix="torch_all"))
    try:
        train_pt_one_step_all(data_path, dump_path)
        pkl_list, npy_list, stack_list = get_pkl_npy_stack_list(
            dump_path, 'torch')
        assert len(pkl_list) == 21
        assert set(pkl_list) == set(npy_list)
        assert len(stack_list) == 7
    finally:
        shutil.rmtree(data_path)
        shutil.rmtree(dump_path)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_api_dump_pt_all_with_scalar():
    data_path = generate_data()
    dump_path = Path(tempfile.mkdtemp(prefix="pt_all_with_scalar"))
    try:
        train_pt_one_step_all(data_path, dump_path, filter_data=False)
        pkl_list, npy_list, stack_list = get_pkl_npy_stack_list(
            dump_path, 'torch')
        assert len(pkl_list) == 25
        assert 'Torch_clip_0_forward_input.1' in pkl_list
        assert 'Torch_clip_0_forward_input.2' in pkl_list
        assert 'Tensor_reshape_0_forward_input.1' in pkl_list
        assert 'Tensor_reshape_0_forward_input.2' in pkl_list
    finally:
        shutil.rmtree(data_path)
        shutil.rmtree(dump_path)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_api_dump_pt_all_with_full_stack():
    data_path = generate_data()
    dump_path = Path(tempfile.mkdtemp(prefix="pt_all_with_full_stack"))
    try:
        train_pt_one_step_all(data_path, dump_path, filter_stack=False)
        pkl_list, npy_list, stack_list = get_pkl_npy_stack_list(
            dump_path, 'torch')
        assert len(stack_list) == 7
    finally:
        shutil.rmtree(data_path)
        shutil.rmtree(dump_path)


def train_pt_one_step_part(data_path, dump_path, info_path=None, retain_backwad=True):
    class Net(BaseNet):
        def forward(self, x):
            api_dump_stop()
            x = self.conv(x)
            api_dump_start()
            x = torch.clip(x, 0.2, 0.5)
            api_dump_stop()
            x = self.bn(x)
            x = self.relu(x)
            x = x.reshape(1, -1)
            x = self.linear(x)
            api_dump_start()
            x = self.relu(x)
            api_dump_stop()
            return x
    train_one_step = BaseTrainOneStep(
        Net(), data_path, dump_path, info_path, retain_backwad)
    train_one_step()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_api_dump_torch_part():
    data_path = generate_data()
    dump_path = Path(tempfile.mkdtemp(prefix="torch_part"))
    try:
        train_pt_one_step_part(data_path, dump_path)
        pkl_list, npy_list, stack_list = get_pkl_npy_stack_list(
            dump_path, 'torch')
        assert len(pkl_list) == 6
        assert set(pkl_list) == set(npy_list)
        assert len(stack_list) == 2
    finally:
        shutil.rmtree(data_path)
        shutil.rmtree(dump_path)


def train_pt_one_step_api_list(data_path, dump_path, info_path=None, retain_backwad=True):
    class Net(BaseNet):
        def forward(self, x):
            api_dump_start(mode='api_list', scope=['relu', 'conv2d'])
            x = self.conv(x)
            x = torch.clip(x, 0.2, 0.5)
            x = self.bn(x)
            x = self.relu(x)
            x = x.reshape(1, -1)
            x = self.linear(x)
            x = self.relu(x)
            return x
    train_one_step = BaseTrainOneStep(
        Net(), data_path, dump_path, info_path, retain_backwad)
    train_one_step()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_api_dump_torch_api_list():
    data_path = generate_data()
    dump_path = Path(tempfile.mkdtemp(prefix="torch_api_list"))
    try:
        train_pt_one_step_api_list(data_path, dump_path)
        pkl_list, npy_list, stack_list = get_pkl_npy_stack_list(
            dump_path, 'torch')

        assert len(pkl_list) == 9
        assert set(pkl_list) == set(npy_list)
        assert len(stack_list) == 3
    finally:
        shutil.rmtree(data_path)
        shutil.rmtree(dump_path)


def train_pt_one_step_list(data_path, dump_path, info_path=None, retain_backwad=True):
    class Net(BaseNet):
        def forward(self, x):
            api_dump_start(mode='list', scope=[
                           'NN_BatchNorm2d_0', 'NN_ReLU_0'])
            x = self.conv(x)
            x = torch.clip(x, 0.2, 0.5)
            x = self.bn(x)
            x = self.relu(x)
            x = x.reshape(1, -1)
            x = self.linear(x)
            x = self.relu(x)
            return x
    train_one_step = BaseTrainOneStep(
        Net(), data_path, dump_path, info_path, retain_backwad)
    train_one_step()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_api_dump_torch_list():
    data_path = generate_data()
    dump_path = Path(tempfile.mkdtemp(prefix="torch_list"))
    try:
        train_pt_one_step_list(data_path, dump_path)
        pkl_list, npy_list, stack_list = get_pkl_npy_stack_list(
            dump_path, 'torch')
        assert len(pkl_list) == 6
        assert set(pkl_list) == set(npy_list)
        assert len(stack_list) == 2
    finally:
        shutil.rmtree(data_path)
        shutil.rmtree(dump_path)


def train_pt_one_step_range(data_path, dump_path, info_path=None, retain_backwad=True):
    class Net(BaseNet):
        def forward(self, x):
            api_dump_start(mode='range', scope=['Torch_clip_0', 'Tensor_reshape_0'])
            x = self.conv(x)
            x = torch.clip(x, 0.2, 0.5)
            x = self.bn(x)
            x = self.relu(x)
            x = x.reshape(1, -1)
            x = self.linear(x)
            x = self.relu(x)
            return x
    train_one_step = BaseTrainOneStep(
        Net(), data_path, dump_path, info_path, retain_backwad)
    train_one_step()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_api_dump_torch_range():
    data_path = generate_data()
    dump_path = Path(tempfile.mkdtemp(prefix="torch_range"))
    try:
        train_pt_one_step_range(data_path, dump_path)
        pkl_list, npy_list, stack_list = get_pkl_npy_stack_list(
            dump_path, 'torch')

        assert len(pkl_list) == 12
        assert set(pkl_list) == set(npy_list)
        assert len(stack_list) == 4
    finally:
        shutil.rmtree(data_path)
        shutil.rmtree(dump_path)
