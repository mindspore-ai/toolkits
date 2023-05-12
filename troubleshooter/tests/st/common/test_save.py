import os
import time
import shutil
import pytest

import troubleshooter as ts
import numpy as np
import torch
import mindspore as ms
from mindspore import nn, Tensor


class NetWorkSave(nn.Cell):
    def __init__(self, file, auto_id, suffix):
        super(NetWorkSave, self).__init__()
        self.auto_id = auto_id
        self.suffix = suffix
        self.file = file

    def construct(self, x):
        ts.save(self.file, x, self.auto_id, self.suffix)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ms_save(mode):
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=mode, device_target="CPU")
    ts.save.cnt.set_data(Tensor(0, ms.int32))
    x1 = Tensor(-0.5962, ms.float32)
    x2 = Tensor(0.4985, ms.float32)
    net = NetWorkSave('/tmp/save/numpy', True, "ms")

    try:
        shutil.rmtree("/tmp/save/")
    except FileNotFoundError:
        pass
    os.makedirs("/tmp/save/")

    x1 = net(x1)
    x2 = net(x2)
    time.sleep(1)
    assert np.allclose(np.load("/tmp/save/0_numpy_ms.npy"),
                       x1.asnumpy())
    assert np.allclose(np.load("/tmp/save/1_numpy_ms.npy"),
                       x2.asnumpy())

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_torch_save(mode):
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=mode, device_target="CPU")
    ts.save.cnt.set_data(Tensor(0, ms.int32))
    x1 = torch.tensor(-0.5962, dtype=torch.float32)
    x2 = torch.tensor(0.4985, dtype=torch.float32)
    file = '/tmp/save/numpy'
    try:
        shutil.rmtree("/tmp/save/")
    except FileNotFoundError:
        pass
    os.makedirs("/tmp/save/")

    ts.save(file, x1, True, "torch")
    ts.save(file, x2, True, "torch")
    time.sleep(1)

    assert np.allclose(np.load("/tmp/save/0_numpy_torch.npy"),
                       x1.cpu().detach().numpy())

    assert np.allclose(np.load("/tmp/save/1_numpy_torch.npy"),
                       x2.cpu().detach().numpy())

@pytest.mark.skip(reason="r2.0 not support")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ms_save_multiple(mode):
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=mode, device_target="CPU")
    ts.save.cnt.set_data(Tensor(0, ms.int32))
    x1 = Tensor(-0.5962, ms.float32)
    x2 = Tensor(0.4985, ms.float32)
    single_input = x1
    list_input = [x1, x2]
    tuple_input = (x2, x1)
    dict_input = {"x1": x1, "x2": x2}
    net = NetWorkSave('/tmp/save/numpy', True, "ms")

    try:
        shutil.rmtree("/tmp/save/")
    except FileNotFoundError:
        pass
    os.makedirs("/tmp/save/")

    single_output = net(single_input)
    list_output = net(list_input)
    tuple_output = net(tuple_input)
    dict_output = net(dict_input)
    time.sleep(1)
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

@pytest.mark.skip(reason="r2.0 not support")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_torch_save_multiple(mode):
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=mode, device_target="CPU")
    ts.save.cnt.set_data(Tensor(0, ms.int32))
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

    ts.save(file, single_input, True, "torch")
    ts.save(file, list_input, True, "torch")
    ts.save(file, tuple_input, True, "torch")
    ts.save(file, dict_input, True, "torch")
    time.sleep(1)

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
