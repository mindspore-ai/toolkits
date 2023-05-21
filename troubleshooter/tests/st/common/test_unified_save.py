import mindspore as ms
import numpy as np
import os
import pytest
import shutil
import time
import torch
from mindspore import nn, Tensor

from troubleshooter.migrator.save import unified_saver


class NetWorkSave(nn.Cell):
    def __init__(self, file):
        super(NetWorkSave, self).__init__()
        self.file = file

    def construct(self, x):
        unified_saver.save(self.file, x)
        return x

class NetWorkSaveMulti(nn.Cell):
    def __init__(self, file, auto_id, suffix):
        super(NetWorkSaveMulti, self).__init__()
        self.file = file
        self.auto_id = auto_id
        self.suffix = suffix

    def construct(self, x):
        unified_saver._save(self.file, x, auto_id=True, suffix="ms")
        return x

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ms_save(mode):
    """
    Feature: unified_saver.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=mode, device_target="CPU")
    unified_saver.save.cnt.set_data(Tensor(0, ms.int32))
    x1 = Tensor(-0.5962, ms.float32)
    x2 = Tensor(0.4985, ms.float32)
    net = NetWorkSave('/tmp/save/numpy_ms')

    try:
        shutil.rmtree("/tmp/save/")
    except FileNotFoundError:
        pass
    os.makedirs("/tmp/save/")

    x1 = net(x1)
    x2 = net(x2)
    time.sleep(0.2)
    assert np.allclose(np.load("/tmp/save/0_numpy_ms.npy"),
                       x1.asnumpy())
    assert np.allclose(np.load("/tmp/save/1_numpy_ms.npy"),
                       x2.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ms_save_none(mode):
    """
    Feature: unified_saver.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=mode, device_target="CPU")
    unified_saver.save.cnt.set_data(Tensor(0, ms.int32))
    x1 = ms.ops.randn((4,))
    x2 = ms.ops.randn((2, 3))
    x3 = ms.ops.randn(tuple())
    net = NetWorkSave(None)

    x1 = net(x1)
    x2 = net(x2)
    x3 = net(x3)
    time.sleep(0.2)
    assert np.allclose(np.load("0_tensor_(4,).npy"),
                       x1.asnumpy())
    assert np.allclose(np.load("1_tensor_(2, 3).npy"),
                       x2.asnumpy())
    assert np.allclose(np.load("2_tensor_().npy"),
                       x3.asnumpy())
    os.remove("0_tensor_(4,).npy")
    os.remove("1_tensor_(2, 3).npy")
    os.remove("2_tensor_().npy")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_torch_save():
    """
    Feature: unified_saver.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    unified_saver.save.cnt.set_data(Tensor(0, ms.int32))
    x1 = torch.tensor(-0.5962, dtype=torch.float32)
    x2 = torch.tensor(0.4985, dtype=torch.float32)
    file = '/tmp/save/numpy_torch'
    try:
        shutil.rmtree("/tmp/save/")
    except FileNotFoundError:
        pass
    os.makedirs("/tmp/save/")

    unified_saver.save(file, x1)
    unified_saver.save(file, x2)
    time.sleep(0.2)

    assert np.allclose(np.load("/tmp/save/0_numpy_torch.npy"),
                       x1.cpu().detach().numpy())

    assert np.allclose(np.load("/tmp/save/1_numpy_torch.npy"),
                       x2.cpu().detach().numpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_torch_save_none():
    """
    Feature: pt_saver.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    unified_saver.save.cnt.set_data(Tensor(0, ms.int32))
    x1 = torch.randn(4)
    x2 = torch.randn(2, 3)
    x3 = torch.randn(tuple())
    file = None

    unified_saver.save(file, x1)
    unified_saver.save(file, x2)
    unified_saver.save(file, x3)
    time.sleep(0.2)

    assert np.allclose(np.load("0_tensor_(4,).npy"),
                       x1.cpu().detach().numpy())

    assert np.allclose(np.load("1_tensor_(2, 3).npy"),
                       x2.cpu().detach().numpy())
    assert np.allclose(np.load("2_tensor_().npy"),
                       x3.cpu().detach().numpy())
    os.remove("0_tensor_(4,).npy")
    os.remove("1_tensor_(2, 3).npy")
    os.remove("2_tensor_().npy")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ms_save_multiple(mode):
    """
    Feature: unified_saver.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=mode, device_target="CPU")
    unified_saver._save.cnt.set_data(Tensor(0, ms.int32))
    x1 = Tensor(-0.5962, ms.float32)
    x2 = Tensor(0.4985, ms.float32)
    single_input = x1
    list_input = [x1, x2]
    tuple_input = (x2, x1)
    dict_input = {"x1": x1, "x2": x2}
    net = NetWorkSaveMulti('/tmp/save/numpy.npy', True, "ms")

    try:
        shutil.rmtree("/tmp/save/")
    except FileNotFoundError:
        pass
    os.makedirs("/tmp/save/")

    single_output = net(single_input)
    list_output = net(list_input)
    tuple_output = net(tuple_input)
    dict_output = net(dict_input)
    time.sleep(0.2)
    assert np.allclose(np.load("/tmp/save/0_numpy_ms.npy"),
                       single_output.asnumpy())

    assert np.allclose(np.load("/tmp/save/1_numpy_0_ms.npy"),
                       list_output[0].asnumpy())
    assert np.allclose(np.load("/tmp/save/1_numpy_1_ms.npy"),
                       list_output[1].asnumpy())

    assert np.allclose(np.load("/tmp/save/2_numpy_0_ms.npy"),
                       tuple_output[0].asnumpy())
    assert np.allclose(np.load("/tmp/save/2_numpy_1_ms.npy"),
                       tuple_output[1].asnumpy())

    assert np.allclose(np.load("/tmp/save/3_numpy_x1_ms.npy"),
                       dict_output["x1"].asnumpy())
    assert np.allclose(np.load("/tmp/save/3_numpy_x2_ms.npy"),
                       dict_output["x2"].asnumpy())



@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_torch_save_multiple():
    """
    Feature: unified_saver.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    unified_saver._save.cnt.set_data(Tensor(0, ms.int32))
    x1 = torch.tensor(-0.5962, dtype=torch.float32)
    x2 = torch.tensor(0.4985, dtype=torch.float32)
    single_input = x1
    list_input = [x1, x2]
    tuple_input = (x2, x1)
    dict_input = {"x1": x1, "x2": x2}
    file = '/tmp/save/numpy'
    try:
        shutil.rmtree("/tmp/save/")
    except FileNotFoundError:
        pass
    os.makedirs("/tmp/save/")

    unified_saver._save(file, single_input, True, "torch")
    unified_saver._save(file, list_input, True, "torch")
    unified_saver._save(file, tuple_input, True, "torch")
    unified_saver._save(file, dict_input, True, "torch")
    time.sleep(0.2)

    assert np.allclose(np.load("/tmp/save/0_numpy_torch.npy"),
                       single_input.cpu().detach().numpy())

    assert np.allclose(np.load("/tmp/save/1_numpy_0_torch.npy"),
                       list_input[0].cpu().detach().numpy())
    assert np.allclose(np.load("/tmp/save/1_numpy_1_torch.npy"),
                       list_input[1].cpu().detach().numpy())

    assert np.allclose(np.load("/tmp/save/2_numpy_0_torch.npy"),
                       tuple_input[0].cpu().detach().numpy())
    assert np.allclose(np.load("/tmp/save/2_numpy_1_torch.npy"),
                       tuple_input[1].cpu().detach().numpy())

    assert np.allclose(np.load("/tmp/save/3_numpy_x1_torch.npy"),
                       dict_input["x1"].cpu().detach().numpy())
    assert np.allclose(np.load("/tmp/save/3_numpy_x2_torch.npy"),
                       dict_input["x2"].cpu().detach().numpy())
