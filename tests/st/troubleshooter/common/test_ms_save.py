import os
import shutil
import time

import mindspore as ms
import numpy as np
import pytest
from mindspore import nn, Tensor
from troubleshooter.migrator.save import mindspore_saver, unified_saver


class NetWorkSave(nn.Cell):
    def __init__(self, saver, file, suffix=None):
        super(NetWorkSave, self).__init__()
        self.saver = saver
        self.file = file
        self.suffix = suffix

    def construct(self, x, auto_id=True):
        self.saver.save(self.file, x, auto_id, self.suffix)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@pytest.mark.parametrize('saver', [mindspore_saver, unified_saver])
def test_ms_save(mode, saver):
    """
    Feature: mindspore_saver.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=mode, device_target="CPU")
    saver.save._clear_cnt()
    x1 = Tensor(-0.5962, ms.float32)
    x2 = Tensor(0.4985, ms.float32)
    single_input = x1
    list_input = [x1, x2]
    tuple_input = (x2, x1)
    dict_input = {"x1": x1, "x2": x2}
    path = f"/tmp/ms_{mode}_{saver.__name__}/"
    net = NetWorkSave(saver, os.path.join(path, 'numpy'), suffix="ms")

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    os.makedirs(path)

    net(single_input, True)
    net(list_input, True)
    net(tuple_input, True)
    net(dict_input, True)
    time.sleep(0.1)

    assert np.allclose(np.load(os.path.join(path, "0_numpy_ms.npy")),
                       single_input.asnumpy())

    assert np.allclose(np.load(os.path.join(path, "1_numpy_0_ms.npy")),
                       list_input[0].asnumpy())
    assert np.allclose(np.load(os.path.join(path, "1_numpy_1_ms.npy")),
                       list_input[1].asnumpy())

    assert np.allclose(np.load(os.path.join(path, "2_numpy_0_ms.npy")),
                       tuple_input[0].asnumpy())
    assert np.allclose(np.load(os.path.join(path, "2_numpy_1_ms.npy")),
                       tuple_input[1].asnumpy())

    assert np.allclose(np.load(os.path.join(path, "3_numpy_x1_ms.npy")),
                       dict_input["x1"].asnumpy())
    assert np.allclose(np.load(os.path.join(path, "3_numpy_x2_ms.npy")),
                       dict_input["x2"].asnumpy())
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@pytest.mark.parametrize('saver', [unified_saver, mindspore_saver])
def test_ms_save_none(mode, saver):
    """
    Feature: unified_saver.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=mode, device_target="CPU")
    saver.save._clear_cnt()
    x0 = ms.ops.randn(tuple())
    x1 = ms.ops.randn((2, 3))
    x2 = [x0, x1]
    x3 = {"x0": x0, "x1": x1}
    file = None
    net = NetWorkSave(saver, file, suffix="ms")
    net(x0, auto_id=False)
    time.sleep(0.1)
    assert np.allclose(np.load("tensor_()_ms.npy"),
                       x0.asnumpy())
    net = NetWorkSave(saver, file)
    net(x0)
    net(x1)
    net(x2)
    net(x3)
    time.sleep(0.1)
    assert np.allclose(np.load("0_tensor_().npy"),
                       x0.asnumpy())
    assert np.allclose(np.load("1_tensor_(2, 3).npy"),
                       x1.asnumpy())
    assert np.allclose(np.load("2_tensor_()_0.npy"),
                       x0.asnumpy())
    assert np.allclose(np.load("2_tensor_(2, 3)_1.npy"),
                       x1.asnumpy())
    assert np.allclose(np.load("3_tensor_()_x0.npy"),
                       x0.asnumpy())
    assert np.allclose(np.load("3_tensor_(2, 3)_x1.npy"),
                       x1.asnumpy())
    net = NetWorkSave(saver, file, suffix="ms")
    net(x3)
    net(x3, auto_id=False)
    time.sleep(0.1)
    assert np.allclose(np.load("4_tensor_()_x0_ms.npy"),
                       x0.asnumpy())
    assert np.allclose(np.load("4_tensor_(2, 3)_x1_ms.npy"),
                       x1.asnumpy())
    assert np.allclose(np.load("tensor_()_x0_ms.npy"),
                       x0.asnumpy())
    assert np.allclose(np.load("tensor_(2, 3)_x1_ms.npy"),
                       x1.asnumpy())

    os.remove("tensor_()_ms.npy")
    os.remove("0_tensor_().npy")
    os.remove("1_tensor_(2, 3).npy")
    os.remove("2_tensor_()_0.npy")
    os.remove("2_tensor_(2, 3)_1.npy")
    os.remove("3_tensor_()_x0.npy")
    os.remove("3_tensor_(2, 3)_x1.npy")
    os.remove("4_tensor_()_x0_ms.npy")
    os.remove("4_tensor_(2, 3)_x1_ms.npy")
    os.remove("tensor_()_x0_ms.npy")
    os.remove("tensor_(2, 3)_x1_ms.npy")
