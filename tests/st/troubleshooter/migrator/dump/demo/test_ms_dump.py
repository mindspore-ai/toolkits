import shutil
import tempfile
from pathlib import Path
import pytest
import troubleshooter as ts
import mindspore as ms
import numpy as np
from mindspore import Tensor, nn, ops

from troubleshooter.migrator import api_dump_init, api_dump_start, api_dump_stop
from tests.st.troubleshooter.migrator.dump.utils import get_pkl_npy_stack_list


def generate_data():
    data_path = Path(tempfile.mkdtemp(prefix="test_data"))
    np.save(data_path / 'label.npy',
            np.random.randn(1, 10).astype(np.float32))
    np.save(data_path / 'data.npy',
            np.random.randn(1, 3, 90, 300).astype(np.float32))
    return data_path


class BaseNet(nn.Cell):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv = nn.Conv2d(3, 5, kernel_size=3, stride=3,
                              padding=0, pad_mode="pad", has_bias=True)
        self.bn = nn.BatchNorm2d(5)
        self.relu = nn.ReLU()
        self.linear = nn.Dense(15000, 10)


class BaseTrainOneStep:
    def __init__(self, net, data_path, dump_path, info_path=None, retain_backward=True) -> None:
        ms.set_context(mode=ms.PYNATIVE_MODE)
        self.net = net
        self.net.set_train()
        if info_path:
            ts.migrator.convert_weight_and_load(weight_map_path=info_path/"torch_net_map.json",
                                                pt_file_path=info_path/"torch_troubleshooter_create.pth",
                                                net=self.net)
        self.data = ms.Tensor(np.load(data_path/'data.npy'))
        self.label = ms.Tensor(np.load(data_path/'label.npy'))
        self.criterion = nn.MSELoss()
        self.optimizer = nn.SGD(
            self.net.trainable_params(), learning_rate=0.01)
        api_dump_init(self.net, dump_path, retain_backward=retain_backward)

    def __call__(self):
        def forward_fn(data, label):
            out = self.net(data)
            loss = self.criterion(out, label)
            return loss
        grad_fn = ms.value_and_grad(
            forward_fn, None, self.optimizer.parameters)
        api_dump_start()
        loss, grads = grad_fn(self.data, self.label)
        self.optimizer(grads)
        api_dump_stop()

def train_ms_one_step_all(data_path, dump_path, info_path=None, retain_backward=True,
                          **api_dump_start_args):
    class Net(BaseNet):
        def construct(self, x):
            api_dump_start(**api_dump_start_args)
            x = self.conv(x)
            x = ops.clip(x, Tensor(0.2, ms.float32), Tensor(0.5, ms.float32))
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
def test_api_dump_ms_all():
    data_path = generate_data()
    dump_path = Path(tempfile.mkdtemp(prefix="ms_all"))
    try:
        train_ms_one_step_all(data_path, dump_path)
        pkl_list, npy_list, stack_list = get_pkl_npy_stack_list(
            dump_path, 'mindspore')
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
def test_api_dump_ms_all_with_scalar():
    data_path = generate_data()
    dump_path = Path(tempfile.mkdtemp(prefix="ms_all_with_scalar"))
    try:
        train_ms_one_step_all(data_path, dump_path, filter_data=False)
        pkl_list, npy_list, stack_list = get_pkl_npy_stack_list(
            dump_path, 'mindspore')
        assert len(pkl_list) == 25
        assert 'Functional_clip_0_forward_input.1' in pkl_list
        assert 'Functional_clip_0_forward_input.2' in pkl_list
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
def test_api_dump_ms_all_with_full_stack():
    data_path = generate_data()
    dump_path = Path(tempfile.mkdtemp(prefix="ms_all_with_full_stack"))
    try:
        train_ms_one_step_all(data_path, dump_path, filter_stack=False)
        pkl_list, npy_list, stack_list = get_pkl_npy_stack_list(
            dump_path, 'mindspore')
        assert len(stack_list) == 7
    finally:
        shutil.rmtree(data_path)
        shutil.rmtree(dump_path)


def train_ms_one_step_part(data_path, dump_path, info_path=None, retain_backwad=True):
    class Net(BaseNet):
        def construct(self, x):
            api_dump_stop()
            x = self.conv(x)
            api_dump_start()
            x = ops.clip(x, Tensor(0.2, ms.float32), Tensor(0.5, ms.float32))
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
def test_api_dump_ms_part():
    data_path = generate_data()
    dump_path = Path(tempfile.mkdtemp(prefix="ms_part"))
    try:
        train_ms_one_step_part(data_path, dump_path)
        pkl_list, npy_list, stack_list = get_pkl_npy_stack_list(
            dump_path, 'mindspore')
        assert len(pkl_list) == 6
        assert set(pkl_list) == set(npy_list)
        assert len(stack_list) == 2
    finally:
        shutil.rmtree(data_path)
        shutil.rmtree(dump_path)


def train_ms_one_step_api_list(data_path, dump_path, info_path=None, retain_backwad=True):
    class Net(BaseNet):
        def construct(self, x):
            api_dump_start(mode='api_list', scope=['relu', 'conv2d'])
            x = self.conv(x)
            x = ops.clip(x, Tensor(0.2, ms.float32), Tensor(0.5, ms.float32))
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
def test_api_dump_ms_api_list():
    data_path = generate_data()
    dump_path = Path(tempfile.mkdtemp(prefix="ms_part"))
    try:
        train_ms_one_step_api_list(data_path, dump_path)
        pkl_list, npy_list, stack_list = get_pkl_npy_stack_list(
            dump_path, 'mindspore')
        assert len(pkl_list) == 9
        assert set(pkl_list) == set(npy_list)
        assert len(stack_list) == 3
    finally:
        shutil.rmtree(data_path)
        shutil.rmtree(dump_path)


def train_ms_one_step_list(data_path, dump_path, info_path=None, retain_backwad=True):
    class Net(BaseNet):
        def construct(self, x):
            api_dump_start(mode='list', scope=[
                           'NN_BatchNorm2d_0', 'NN_ReLU_0'])
            x = self.conv(x)
            x = ops.clip(x, 0.2, 0.5)
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
def test_api_dump_ms_list():
    data_path = generate_data()
    dump_path = Path(tempfile.mkdtemp(prefix="ms_list"))
    try:
        train_ms_one_step_list(data_path, dump_path)
        pkl_list, npy_list, stack_list = get_pkl_npy_stack_list(
            dump_path, 'mindspore')
        assert len(pkl_list) == 6
        assert set(pkl_list) == set(npy_list)
        assert len(stack_list) == 2
    finally:
        shutil.rmtree(data_path)
        shutil.rmtree(dump_path)


def train_ms_one_step_range(data_path, dump_path, info_path=None, retain_backwad=True):
    class Net(BaseNet):
        def construct(self, x):
            api_dump_start(mode='range', scope=[
                           'Functional_clip_0', 'Tensor_reshape_0'])
            x = self.conv(x)
            x = ops.clip(x, 0.2, 0.5)
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
def test_api_dump_ms_range():
    data_path = generate_data()
    dump_path = Path(tempfile.mkdtemp(prefix="ms_range"))
    try:
        train_ms_one_step_range(data_path, dump_path)
        pkl_list, npy_list, stack_list = get_pkl_npy_stack_list(
            dump_path, 'mindspore')
        assert len(pkl_list) == 12
        assert set(pkl_list) == set(npy_list)
        assert len(stack_list) == 4
    finally:
        shutil.rmtree(data_path)
        shutil.rmtree(dump_path)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_api_dump_ms_with_not_float_output():
    x = Tensor(np.random.randn(8, 5).astype(np.float32))
    dump_path = Path(tempfile.mkdtemp(prefix="with_not_float_output"))
    ts.migrator.api_dump_init(ms.nn.Cell(), dump_path, retain_backward=True)
    ts.migrator.api_dump_start()
    out = x.max(axis=1, return_indices=True)
    ts.migrator.api_dump_stop()
    shutil.rmtree(dump_path)
    assert len(out) == 2
