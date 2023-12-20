from pathlib import Path

import numpy as np

import troubleshooter as ts
import mindspore as ms
import torch
import tempfile
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_compare_list_dir(capsys):
    dir1 = tempfile.TemporaryDirectory()
    dir2 = tempfile.TemporaryDirectory()
    path1 = Path(dir1.name)
    path2 = Path(dir2.name)
    a1 = np.random.rand(1, 3, 2).astype(np.float32)
    a2 = np.random.rand(1, 3, 2).astype(np.float32)
    data1 = [ms.Tensor(a1), ms.Tensor(a2)]
    ts.save(str(path1 / "data"), data1)
    data2 = [torch.Tensor(a1), torch.Tensor(a2)]
    ts.save(str(path2 / "data"), data2)

    ts.migrator.compare_list_dir(path1, path2)
    result = capsys.readouterr().out
    assert result.count('True') == 2
