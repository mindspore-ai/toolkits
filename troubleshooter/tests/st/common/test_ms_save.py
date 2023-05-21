import mindspore as ms
import numpy as np
import os
import pytest
import shutil
import time
from mindspore import nn, Tensor

from troubleshooter.migrator.save import mindspore_saver


class NetWorkSave(nn.Cell):
    def __init__(self, file):
        super(NetWorkSave, self).__init__()
        self.file = file

    def construct(self, x):
        mindspore_saver.save(self.file, x)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ms_save(mode):
    """
    Feature: mindspore_saver.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=mode, device_target="CPU")
    mindspore_saver.save.cnt.set_data(Tensor(0, ms.int32))
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
    Feature: mindspore_saver.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=mode, device_target="CPU")
    mindspore_saver.save.cnt.set_data(Tensor(0, ms.int32))
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
