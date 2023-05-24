import numpy as np
import os
import shutil
import troubleshooter as ts


def test_compare_npy_dir(capsys):
    data1 = np.random.rand(1, 3, 2).astype(np.float32)
    data2 = np.random.rand(1, 3, 2).astype(np.float32)

    path1 = "/tmp/troubleshooter_ta/"
    path2 = "/tmp/troubleshooter_tb/"
    isexists  =os.path.exists(path1)
    if not isexists:
        os.makedirs(path1)

    isexists = os.path.exists(path2)
    if not isexists:
        os.makedirs(path2)

    np.save('/tmp/troubleshooter_ta/data1.npy',data1)
    np.save('/tmp/troubleshooter_ta/data2.npy',data2)

    np.save('/tmp/troubleshooter_tb/data1.npy',data1)
    np.save('/tmp/troubleshooter_tb/data2.npy',data1)

    ts.diff_handler.compare_npy_dir(path1, path2)
    result = capsys.readouterr().out

    shutil.rmtree(path1)
    shutil.rmtree(path2)
    assert result.count('True') == 1
