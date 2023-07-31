import os
import shutil
import time
import tempfile

import mindspore as ms
import numpy as np
import pytest
from mindspore import nn, ops
from troubleshooter.migrator.save import _ts_save_cnt

import troubleshooter as ts


class NetWorkSave(nn.Cell):
    def __init__(self, file, suffix=None):
        super(NetWorkSave, self).__init__()
        self.file = file
        self.suffix = suffix

    def construct(self, x, auto_id=True):
        ts.save(self.file, x, auto_id, self.suffix)
        out = ops.clip_by_value(x, clip_value_min=0.2, clip_value_max=0.8)
        ts.save(self.file, out, auto_id, self.suffix)
        return out


class NetWorkSaveSimple(nn.Cell):
    def __init__(self, file, suffix=None):
        super(NetWorkSaveSimple, self).__init__()
        self.file = file
        self.suffix = suffix

    def construct(self, x, auto_id=True):
        ts.save(self.file, x, auto_id, self.suffix)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ms_save_single(mode):
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=mode)
    _ts_save_cnt.reset()
    single_input = ms.ops.randn((2, 3))
    path = f"/tmp/save_ms_single_{mode}/"
    net = NetWorkSave(os.path.join(path, 'numpy'), suffix="ms")

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    os.makedirs(path)

    out = net(single_input, True)
    time.sleep(0.1)

    assert np.allclose(np.load(os.path.join(path, "0_numpy_ms.npy")),
                       single_input.asnumpy())
    assert np.allclose(np.load(os.path.join(path, "1_numpy_ms.npy")),
                       out.asnumpy())
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ms_save_iter(mode):
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=mode)
    _ts_save_cnt.reset()
    x1 = ops.randn((3, 5))
    x2 = ops.randn((3, 4))
    list_input = [x1, x2]
    tuple_input = (x2, x1)
    path = f"/tmp/save_ms_iter_{mode}/"
    net = NetWorkSave(os.path.join(path, 'numpy'), suffix="ms")

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    os.makedirs(path)

    out0 = net(list_input, True)

    out1 = net(tuple_input, True)
    time.sleep(0.1)

    assert np.allclose(np.load(os.path.join(path, "0_numpy_0_ms.npy")),
                       list_input[0].asnumpy())
    assert np.allclose(np.load(os.path.join(path, "0_numpy_1_ms.npy")),
                       list_input[1].asnumpy())

    assert np.allclose(np.load(os.path.join(path, "1_numpy_0_ms.npy")),
                       out0[0].asnumpy())
    assert np.allclose(np.load(os.path.join(path, "1_numpy_1_ms.npy")),
                       out0[1].asnumpy())

    assert np.allclose(np.load(os.path.join(path, "2_numpy_0_ms.npy")),
                       tuple_input[0].asnumpy())
    assert np.allclose(np.load(os.path.join(path, "2_numpy_1_ms.npy")),
                       tuple_input[1].asnumpy())

    assert np.allclose(np.load(os.path.join(path, "3_numpy_0_ms.npy")),
                       out1[0].asnumpy())
    assert np.allclose(np.load(os.path.join(path, "3_numpy_1_ms.npy")),
                       out1[1].asnumpy())

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ms_save_dict(mode):
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=mode)
    _ts_save_cnt.reset()
    x1 = ops.randn((3, 5))
    x2 = ops.randn((3, 4))
    dict_input = {"x1": x1, "x2": x2}
    path = f"/tmp/save_ms_dict_{mode}/"
    net = NetWorkSaveSimple(os.path.join(path, 'numpy'), suffix="ms")

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    os.makedirs(path)

    net(dict_input, True)
    time.sleep(0.1)

    assert np.allclose(np.load(os.path.join(path, "0_numpy_x1_ms.npy")),
                       dict_input["x1"].asnumpy())
    assert np.allclose(np.load(os.path.join(path, "0_numpy_x2_ms.npy")),
                       dict_input["x2"].asnumpy())
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ms_save_none(mode):
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=mode)
    _ts_save_cnt.reset()
    x0 = ms.ops.randn(tuple())
    x1 = ms.ops.randn((2, 3))
    x2 = [x0, x1]
    x3 = {"x0": x0, "x1": x1}
    file = None
    net = NetWorkSaveSimple(file, suffix="ms")
    net(x0, auto_id=False)
    time.sleep(0.1)
    assert np.allclose(np.load("tensor_()_ms.npy"),
                       x0.asnumpy())
    net = NetWorkSaveSimple(file)
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
    net = NetWorkSaveSimple(file, suffix="ms")
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_ms_list_with_none(mode):
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    _ts_save_cnt.reset()
    ms.set_context(mode=mode)
    x0 = ms.ops.randn(tuple())
    x3 = {"x0": x0, "x1": None}
    path = tempfile.mkdtemp(prefix="ms_list_with_none")
    file = os.path.join(path, "list_with_none")

    ts.save(file, x3)
    time.sleep(0.1)
    assert np.allclose(np.load(os.path.join(path, "0_list_with_none_x0.npy")),
                       x0.asnumpy())

    shutil.rmtree(path)
