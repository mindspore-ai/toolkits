import time
import numpy as np
import pytest
import troubleshooter as ts

from tempfile import TemporaryDirectory
from pathlib import Path


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('file', ['./test_files/vm_format_print_file', './test_files/ge_str_format_print_file'])
def test_save_convert(file):
    """
    Feature: ts.widget.save_convert
    Description: Verify the result of save_convert
    Expectation: success
    """
    tmp_dir = TemporaryDirectory()
    path = Path(tmp_dir.name) / "multi" / "dir"
    ts.widget.save_convert(file, path)
    time.sleep(0.1)
    except_a = np.array([2, 3, 4, 5], dtype=np.float32)
    except_b = np.array([4, 5, 6, 7], dtype=np.float32)
    except_c = np.array([20, 25, 30, 35], dtype=np.float32)
    a = np.load(path/'add_float32_0.npy')
    b = np.load(path/'add_float32_1.npy')
    c = np.load(path/'mul_float32_2.npy')
    assert np.allclose(a, except_a)
    assert np.allclose(b, except_b)
    assert np.allclose(c, except_c)
