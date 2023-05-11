import numpy as np


def test_compare_npy_dir():
    data1 = np.random.rand(1, 3, 22, 33).astype(np.float32)
    data2 = np.random.rand(1, 3, 22, 33).astype(np.float32)
    data3 = np.random.rand(1, 3, 22, 33).astype(np.float32)
    data4 = np.random.rand(1, 3, 22, 33).astype(np.float32)

    #np.save('./tta/data1.npy',data1)
    #np.save('./ta/data2.npy',data2)
    #np.save('./ta/data3.npy',data3)
    #np.save('./ta/data4.npy',data4)

