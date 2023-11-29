import tempfile
from pathlib import Path

import numpy as np
import pytest
import troubleshooter as ts

from tests.util import check_delimited_list


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("a, b",
                         [(np.random.rand(1, 3, 2).astype(np.float32), np.random.rand(1, 3, 2).astype(np.float32)),
                          (np.array([True, False, False, False]), np.array([True, False, True, True])),
                          (np.array([1, 2, 3, 4]), np.array([1, 2, 4, 3])), (np.float32(np.inf), np.float32(2))])
def test_compare_npy_dir_should_all_equal_with_multi_data_type(capsys, a, b):
    dir1 = tempfile.TemporaryDirectory()
    dir2 = tempfile.TemporaryDirectory()
    path1 = Path(dir1.name)
    path2 = Path(dir2.name)

    np.save(str(path1 / "data1.npy"), a)
    np.save(str(path1 / "data2.npy"), b)

    np.save(str(path2 / "data1.npy"), a)
    np.save(str(path2 / "data2.npy"), b)

    ts.migrator.compare_npy_dir(path1, path2)
    result = capsys.readouterr().out
    assert result.count('True') == 2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("a, b",
                         [(np.random.rand(1, 3, 2).astype(np.float32), np.random.rand(1, 3, 2).astype(np.float32)),
                          (np.array([True, False, False, False]), np.array([True, False, True, True])),
                          (np.array([1, 2, 3, 4]), np.array([1, 2, 4, 3])), (np.float32(np.inf), np.float32(2))])
def test_compare_npy_dir_part_equal_with_multi_data_type(capsys, a, b):
    dir1 = tempfile.TemporaryDirectory()
    dir2 = tempfile.TemporaryDirectory()
    path1 = Path(dir1.name)
    path2 = Path(dir2.name)

    np.save(str(path1 / "data1.npy"), a)
    np.save(str(path1 / "data2.npy"), b)

    np.save(str(path2 / "data1.npy"), a)
    np.save(str(path2 / "data2.npy"), a)

    ts.migrator.compare_npy_dir(path1, path2)
    result = capsys.readouterr().out
    assert result.count('True') == 1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_compare_npy_dir_scalar(capsys):
    dir1 = tempfile.TemporaryDirectory()
    dir2 = tempfile.TemporaryDirectory()
    path1 = Path(dir1.name)
    path2 = Path(dir2.name)

    a = np.float32(np.inf)
    b = np.float32(2)
    c = np.float32(1)
    np.save(str(path1 / "data1.npy"), a)
    np.save(str(path1 / "data2.npy"), b)

    np.save(str(path2 / "data1.npy"), a)
    np.save(str(path2 / "data2.npy"), c)

    ts.migrator.compare_npy_dir(path1, path2)
    result = capsys.readouterr().out

    assert check_delimited_list(result, ["data1.npy", "data1.npy", "True", "100.00%", "1.00000", "0.00000, 0.00000"])
    assert check_delimited_list(result, ["data2.npy", "data2.npy", "False", "0.00%", "1.00000", "1.00000, 1.00000"])

