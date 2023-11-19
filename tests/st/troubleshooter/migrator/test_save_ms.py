import time
import tempfile
from pathlib import Path

import mindspore as ms
import numpy as np
import pytest
from mindspore import nn, ops

import troubleshooter as ts
from troubleshooter.common.util import find_file, extract_end_number


class NetWorkSave(nn.Cell):
    def __init__(self, file, suffix=None):
        super(NetWorkSave, self).__init__()
        self.file = file
        self.suffix = suffix

    def construct(self, x):
        out = ops.clip_by_value(x, clip_value_min=0.2, clip_value_max=0.8)
        ts.save(self.file, out, self.suffix)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ms_save_single(mode):
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=mode)
    single_input = ms.ops.randn((2, 3))
    dir = tempfile.TemporaryDirectory(prefix="save_ms_single")
    path = Path(dir.name)
    net = NetWorkSave(str(path / "numpy"), suffix="ms")

    out = net(single_input)
    time.sleep(0.1)

    file_list = find_file(path)
    print(file_list)
    assert len(file_list) == 1
    assert np.allclose(np.load(path / file_list[0]), out.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [ ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ms_save_iter(mode):
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=mode, save_graphs=False, save_graphs_path="./iter")
    x1 = ops.randn((3, 5))
    x2 = ops.randn((3, 5))
    list_input = [x1, x2]
    dir = tempfile.TemporaryDirectory(prefix="save_ms_iter")
    path = Path(dir.name)
    file = str(path / "numpy")
    net = NetWorkSave(file, suffix="ms")

    out0 = net(list_input)
    time.sleep(0.1)
    file_list = find_file(path, sort_key=extract_end_number)
    print(file_list)
    assert len(file_list) == len(out0)
    assert np.allclose(np.load(path / file_list[0]), out0[0].asnumpy())
    assert np.allclose(np.load(path / file_list[1]), out0[1].asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ms_save_dict(mode):
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    class NetWorkSaveSimple(nn.Cell):
        def __init__(self, file, suffix=None):
            super(NetWorkSaveSimple, self).__init__()
            self.file = file
            self.suffix = suffix
            self.d = {}

        def construct(self, x1, x2):
            self.d['x1'] = x1 + 0.1
            self.d['x2'] = x2 - 0.1
            ts.save(self.file, self.d, self.suffix)
            return self.d

    ms.set_context(mode=mode)
    x1 = ops.randn((3, 5))
    x2 = ops.randn((3, 4))
    dir = tempfile.TemporaryDirectory(prefix="save_ms_dict")
    path = Path(dir.name)
    file = str(path / "numpy")
    net = NetWorkSaveSimple(file, suffix="ms")

    net(x1, x2)
    time.sleep(0.1)
    file_list = find_file(path, sort_key=extract_end_number)
    print(file_list)
    assert len(file_list) == 2
    assert np.allclose(np.load(path / file_list[0]), x1.asnumpy() + 0.1)
    assert np.allclose(np.load(path / file_list[1]), x2.asnumpy() - 0.1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [ms.PYNATIVE_MODE])
def test_save_multi_level_input(mode):
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    class NetWorkSaveSimple(nn.Cell):
        def __init__(self, file, suffix=None):
            super(NetWorkSaveSimple, self).__init__()
            self.file = file
            self.suffix = suffix
            self.d = {}

        def construct(self, x1, x2):
            self.d['x1'] = x1 + 0.1
            self.d['x2'] = x2 - 0.1
            t = [self.d, x2 - 0.3]
            ts.save(self.file, t, self.suffix)
            return t
    ms.set_context(mode=mode)
    x1 = ops.randn((3, 5))
    x2 = ops.randn((3, 4))
    dir = tempfile.TemporaryDirectory(prefix="save_ms_multi_level")
    path = Path(dir.name)
    file = str(path / "numpy")
    net = NetWorkSaveSimple(file)
    net(x1, x2)
    time.sleep(0.1)
    file_list = find_file(path)
    print(file_list)
    assert len(file_list) == 3
    assert np.allclose(np.load(path / file_list[0]), x1.asnumpy() + 0.1)
    assert np.allclose(np.load(path / file_list[1]), x2.asnumpy() - 0.1)
    assert np.allclose(np.load(path / file_list[2]), x2.asnumpy() - 0.3)
