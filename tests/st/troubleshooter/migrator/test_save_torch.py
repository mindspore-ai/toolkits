import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import troubleshooter as ts
from troubleshooter.common.util import find_file


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_torch_save_single():
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    single_input = torch.randn((2, 3))
    dir = tempfile.TemporaryDirectory(prefix="save_torch_single")
    path = Path(dir.name)

    ts.save(str(path / "numpy"), single_input, suffix="torch")
    time.sleep(0.1)
    file_list = find_file(path)
    assert len(file_list) == 1
    assert np.allclose(
        np.load(path / file_list[0]), single_input.cpu().detach().numpy()
    )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_torch_save_iter():
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    x1 = torch.randn((3, 5))
    x2 = torch.randn((3, 4))
    list_input = [x1, x2]
    tuple_input = (x2, x1)
    dir = tempfile.TemporaryDirectory(prefix="save_torch_iter")
    path = Path(dir.name)
    file = str(path / "numpy")

    ts.save(file, list_input, suffix="torch")
    ts.save(file, tuple_input, suffix="torch")

    time.sleep(0.1)
    file_list = find_file(path)
    assert len(file_list) == len(list_input) + len(tuple_input)
    assert np.allclose(
        np.load(path / file_list[0]), list_input[0].cpu().detach().numpy()
    )
    assert np.allclose(
        np.load(path / file_list[1]), list_input[1].cpu().detach().numpy()
    )

    assert np.allclose(
        np.load(path / file_list[2]), tuple_input[0].cpu().detach().numpy()
    )
    assert np.allclose(
        np.load(path / file_list[3]), tuple_input[1].cpu().detach().numpy()
    )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_torch_save_dict():
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    x1 = torch.randn((3, 5))
    x2 = torch.randn((3, 4))
    dict_input = {"x1": x1, "x2": x2}
    dir = tempfile.TemporaryDirectory(prefix="save_torch_dict")
    path = Path(dir.name)
    file = str(path / "numpy")

    ts.save(file, dict_input, suffix="torch")

    time.sleep(0.1)
    file_list = find_file(path)
    assert len(file_list) == len(dict_input)
    assert np.allclose(np.load(path / file_list[0]), x1.cpu().detach().numpy())
    assert np.allclose(np.load(path / file_list[1]), x2.cpu().detach().numpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_save_multi_level_input():
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    x1 = torch.randn((3, 5))
    x2 = torch.randn((3, 4))
    multi_level_input = {"d1": [x1, x2], "d2": (x1,)}
    dir = tempfile.TemporaryDirectory(prefix="save_torch_multi")
    path = Path(dir.name)
    file = str(path / "numpy")
    ts.save(file, multi_level_input)
    time.sleep(0.1)
    file_list = find_file(path)
    print(file_list)
    assert len(file_list) == 3
    assert np.allclose(np.load(path / file_list[0]), x1.cpu().detach().numpy())
    assert np.allclose(np.load(path / file_list[1]), x2.cpu().detach().numpy())
    assert np.allclose(np.load(path / file_list[2]), x1.cpu().detach().numpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_torch_list_with_none():
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    x0 = torch.randn(tuple())
    x3 = {"x0": x0, "x1": None}
    dir = tempfile.TemporaryDirectory(prefix="torch_list_with_none")
    path = Path(dir.name)
    file = str(path / "list_with_none")

    ts.save(file, x3)
    time.sleep(0.1)
    file_list = find_file(path)
    assert np.allclose(np.load(path / file_list[0]), x0.cpu().detach().numpy())
