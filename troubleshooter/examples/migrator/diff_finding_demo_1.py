import numpy as np
import troubleshooter as ts

if __name__ == '__main__':
    #name = {"data1.npy":"data1.npy","data2.npy":"data2.npy","data3.npy":"data3.npy","data4.npy":"data4.npy"}
    #ta = "/mnt/d/06_project/troubleshooter/troubleshooter/tests/diff_handler/ta"
    #tb = "/mnt/d/06_project/troubleshooter/troubleshooter/tests/diff_handler/tb"
    ta = "/mnt/d/06_project/troubleshooter/troubleshooter/examples/migrator/save_tensor/resnet_ms/ms1"
    tb = "/mnt/d/06_project/troubleshooter/troubleshooter/examples/migrator/save_tensor/resnet_pytorch/pt1"
    dif = ts.diff_finder(ta, tb)
    dif.compare_npy_dir()

    #name_list = dif.get_filename_map_list()
    #dif.compare_npy_dir(name_map_list=name_list)
