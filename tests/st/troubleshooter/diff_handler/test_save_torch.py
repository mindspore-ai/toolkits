import os
import shutil
import time

import numpy as np
import pytest
import torch
from troubleshooter.migrator.save import _ts_save_cnt

import troubleshooter as ts


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
    _ts_save_cnt.reset()
    single_input = torch.randn((2, 3))
    path = f"/tmp/save_torch_single/"

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    os.makedirs(path)

    ts.save(os.path.join(path, "numpy"), single_input, suffix="torch")
    time.sleep(0.1)

    assert np.allclose(np.load(os.path.join(path, "0_numpy_torch.npy")),
                       single_input.cpu().detach().numpy())
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
def test_torch_save_iter():
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    _ts_save_cnt.reset()
    x1 = torch.randn((3, 5))
    x2 = torch.randn((3, 4))
    list_input = [x1, x2]
    tuple_input = (x2, x1)
    path = f"/tmp/save_torch_iter/"
    file = os.path.join(path, "numpy")

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    os.makedirs(path)

    ts.save(file, list_input, suffix="torch")
    ts.save(file, tuple_input, suffix="torch")

    time.sleep(0.1)

    assert np.allclose(np.load(os.path.join(path, "0_numpy_0_torch.npy")),
                       list_input[0].cpu().detach().numpy())
    assert np.allclose(np.load(os.path.join(path, "0_numpy_1_torch.npy")),
                       list_input[1].cpu().detach().numpy())

    assert np.allclose(np.load(os.path.join(path, "1_numpy_0_torch.npy")),
                       tuple_input[0].cpu().detach().numpy())
    assert np.allclose(np.load(os.path.join(path, "1_numpy_1_torch.npy")),
                       tuple_input[1].cpu().detach().numpy())

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
def test_torch_save_dict():
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    _ts_save_cnt.reset()
    x1 = torch.randn((3, 5))
    x2 = torch.randn((3, 4))
    dict_input = {"x1": x1, "x2": x2}
    path = f"/tmp/save_torch_dict/"
    file = os.path.join(path, "numpy")
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    os.makedirs(path)

    ts.save(file, dict_input, suffix="torch")

    time.sleep(0.1)
    assert np.allclose(np.load(os.path.join(path, "0_numpy_x1_torch.npy")),
                       dict_input["x1"].cpu().detach().numpy())
    assert np.allclose(np.load(os.path.join(path, "0_numpy_x2_torch.npy")),
                       dict_input["x2"].cpu().detach().numpy())
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
def test_torch_save_none():
    """
    Feature: ts.save
    Description: Verify the result of save
    Expectation: success
    """
    _ts_save_cnt.reset()
    x0 = torch.randn(tuple())
    x1 = torch.randn(2, 3)
    x2 = [x0, x1]
    x3 = {"x0": x0, "x1": x1}
    file = None

    ts.save(file, x0, auto_id=False, suffix="torch")
    ts.save(file, x0)
    ts.save(file, x1)
    ts.save(file, x2)
    ts.save(file, x3)
    ts.save(file, x3, suffix="torch")
    ts.save(file, x3, auto_id=False, suffix="torch")
    time.sleep(0.1)
    # single auto_id=False, suffix=torch
    assert np.allclose(np.load("tensor_()_torch.npy"),
                       x0.cpu().detach().numpy())
    # single auto_id=True
    assert np.allclose(np.load("0_tensor_().npy"),
                       x0.cpu().detach().numpy())
    # multi auto_id=True
    assert np.allclose(np.load("1_tensor_(2, 3).npy"),
                       x1.cpu().detach().numpy())
    assert np.allclose(np.load("2_tensor_()_0.npy"),
                       x0.cpu().detach().numpy())
    assert np.allclose(np.load("2_tensor_(2, 3)_1.npy"),
                       x1.cpu().detach().numpy())
    assert np.allclose(np.load("3_tensor_()_x0.npy"),
                       x0.cpu().detach().numpy())
    assert np.allclose(np.load("3_tensor_(2, 3)_x1.npy"),
                       x1.cpu().detach().numpy())
    assert np.allclose(np.load("4_tensor_()_x0_torch.npy"),
                       x0.cpu().detach().numpy())
    assert np.allclose(np.load("4_tensor_(2, 3)_x1_torch.npy"),
                       x1.cpu().detach().numpy())
    assert np.allclose(np.load("tensor_()_x0_torch.npy"),
                       x0.cpu().detach().numpy())
    assert np.allclose(np.load("tensor_(2, 3)_x1_torch.npy"),
                       x1.cpu().detach().numpy())

    os.remove("tensor_()_torch.npy")
    os.remove("0_tensor_().npy")
    os.remove("1_tensor_(2, 3).npy")
    os.remove("2_tensor_()_0.npy")
    os.remove("2_tensor_(2, 3)_1.npy")
    os.remove("3_tensor_()_x0.npy")
    os.remove("3_tensor_(2, 3)_x1.npy")
    os.remove("4_tensor_()_x0_torch.npy")
    os.remove("4_tensor_(2, 3)_x1_torch.npy")
    os.remove("tensor_()_x0_torch.npy")
    os.remove("tensor_(2, 3)_x1_torch.npy")
