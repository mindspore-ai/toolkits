import numpy as np
import os
import pytest
import shutil
import time
import torch

from troubleshooter.migrator.save import pt_saver


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_torch_save():
    """
    Feature: pt_saver.save
    Description: Verify the result of save
    Expectation: success
    """
    pt_saver._g_save_cnt = 0
    x1 = torch.tensor(-0.5962, dtype=torch.float32)
    x2 = torch.tensor(0.4985, dtype=torch.float32)
    file = '/tmp/save/numpy_torch'
    try:
        shutil.rmtree("/tmp/save/")
    except FileNotFoundError:
        pass
    os.makedirs("/tmp/save/")

    pt_saver.save(file, x1)
    pt_saver.save(file, x2)
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
    pt_saver._g_save_cnt = 0
    x1 = torch.randn(4)
    x2 = torch.randn(2, 3)
    x3 = torch.randn(tuple())
    file = None

    pt_saver.save(file, x1)
    pt_saver.save(file, x2)
    pt_saver.save(file, x3)
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
