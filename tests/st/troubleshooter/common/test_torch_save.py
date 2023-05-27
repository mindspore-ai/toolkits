import os
import shutil
import time

import mindspore as ms
import numpy as np
import pytest
import torch
from troubleshooter.migrator.save import torch_saver, unified_saver


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('saver', [torch_saver, unified_saver])
def test_torch_save(saver):
    """
    Feature: unified_saver.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    saver.save._clear_cnt()
    x1 = torch.tensor(-0.5962, dtype=torch.float32)
    x2 = torch.tensor(0.4985, dtype=torch.float32)
    single_input = x1
    list_input = [x1, x2]
    tuple_input = (x2, x1)
    dict_input = {"x1": x1, "x2": x2}
    path = f"/tmp/pt_{saver.__name__}"
    file = os.path.join(path, "numpy")
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    os.makedirs(path)

    saver.save(file, single_input, True, "torch")
    saver.save(file, list_input, True, "torch")
    saver.save(file, tuple_input, True, "torch")
    saver.save(file, dict_input, True, "torch")
    time.sleep(0.2)

    assert np.allclose(np.load(os.path.join(path, "0_numpy_torch.npy")),
                       single_input.cpu().detach().numpy())

    assert np.allclose(np.load(os.path.join(path, "1_numpy_0_torch.npy")),
                       list_input[0].cpu().detach().numpy())
    assert np.allclose(np.load(os.path.join(path, "1_numpy_1_torch.npy")),
                       list_input[1].cpu().detach().numpy())

    assert np.allclose(np.load(os.path.join(path, "2_numpy_0_torch.npy")),
                       tuple_input[0].cpu().detach().numpy())
    assert np.allclose(np.load(os.path.join(path, "2_numpy_1_torch.npy")),
                       tuple_input[1].cpu().detach().numpy())

    assert np.allclose(np.load(os.path.join(path, "3_numpy_x1_torch.npy")),
                       dict_input["x1"].cpu().detach().numpy())
    assert np.allclose(np.load(os.path.join(path, "3_numpy_x2_torch.npy")),
                       dict_input["x2"].cpu().detach().numpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
@pytest.mark.parametrize('saver', [torch_saver, unified_saver])
def test_torch_save_none(saver):
    """
    Feature: pt_saver.save
    Description: Verify the result of save
    Expectation: success
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    saver.save._clear_cnt()
    x0 = torch.randn(tuple())
    x1 = torch.randn(2, 3)
    x2 = [x0, x1]
    x3 = {"x0": x0, "x1": x1}
    file = None

    saver.save(file, x0, auto_id=False, suffix="torch")
    saver.save(file, x0)
    saver.save(file, x1)
    saver.save(file, x2)
    saver.save(file, x3)
    saver.save(file, x3, suffix="torch")
    saver.save(file, x3, auto_id=False, suffix="torch")
    time.sleep(0.2)
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
