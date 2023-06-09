#!/usr/bin/env python3
# coding=utf-8

"""
This is a python code template
"""
from mindspore import Tensor, nn, ops
import mindspore
import numpy as np

import troubleshooter as ts
from tests.util import delete_file, file_and_key_match


def test_ascend_e39999():
    @ts.proposal(write_file_path="/tmp/")
    def test():
        raise ValueError("E39999: Inner Error!")

    delete_file("/tmp/")
    test()
    assert file_and_key_match("/tmp/", "vm_id_3")


def test_ascend_ee8888():
    @ts.proposal(write_file_path="/tmp/")
    def test():
        raise ValueError("EE8888: Inner Error!")

    delete_file("/tmp/")
    test()
    result = file_and_key_match("/tmp/", "vm_id_1")
    assert result


def test_ascend_ej0001():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        raise ValueError("EJ0001: XXXXX")

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "vm_id_8")


def test_ascend_ei0002():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        raise ValueError("EI0002: The wait execution of the Notify register times out")

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "vm_id_5")


def test_ascend_ei0004():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        raise ValueError("EI0004: The ranktable is invalid")

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "vm_id_4")


def test_ascend_ei0004_2():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        error = """
        Traceback (most recent call last):
          File "train.py", line 380, in <module>
            train_net()
          File "/opt/nvme2n1/m00256308/models-master/official/cv/resnet/scripts/train_parallel6/src/model_utils/moxing_adapter.py", line 104, in wrapped_func
            run_func(*args, **kwargs)
          File "train.py", line 307, in train_net
            set_parameter()
          File "train.py", line 154, in set_parameter
            init()
          File "/opt/nvme0n1/root/miniforge3/envs/miaoym/lib/python3.7/site-packages/mindspore/communication/management.py", line 151, in init
            init_hccl()
        RuntimeError: Ascend collective communication initialization failed.
    
        ----------------------------------------------------
        - Ascend Error Message:
        ----------------------------------------------------
        EI0004: The ranktable is invalid,Reason:[Use a rank ID that exceeds the rank size in the ranktable.]. Please check the configured ranktable. [{"server_count":"1","server_list":[{"device":[{"device_id":"6","device_ip":"192.168.102.112","rank_id":"0"},{"device_id":"7","device_ip":"192.168.103.112","rank_id":"1"}],"host_nic_ip":"reserve","server_id":"10.90.42.160"}],"status":"completed","version":"1.0"}]
                Solution: Try again with a valid cluster configuration in the ranktable file. Ensure that the configuration matches the operating environment.
    
        (Please search "Ascend Error Message" at https://www.mindspore.cn for error code description)
    
        ----------------------------------------------------
        - Framework Error Message: (For framework developers)
        ----------------------------------------------------
        Init hccl graph adapter failed.
    
        ----------------------------------------------------
        - C++ Call Stack: (For framework developers)
        ----------------------------------------------------
        mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.cc:112 Initialize
        mindspore/ccsrc/plugin/device/ascend/hal/hccl_adapter/hccl_adapter.cc:402 InitKernelInfoStore
               """
        raise ValueError(error)

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "vm_id_4")


def test_ascend_ej0006():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        raise ValueError("EI0006: xxx")

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "vm_id_6")


def test_ascend_ei9999():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        raise ValueError("EI9999: XXXX")

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "vm_id_7")


def test_ascend_malloc_device_memory_failed():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        raise ValueError("Malloc device memory failed, size[32212254720], ret[207001], Device 6 may be other processes "
                         "occupying this card, check as: ps -ef|grep python")

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "vm_id_2")


def test_ascend_run_task_error_ez9999():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        raise ValueError("run task error")

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "vm_g_id_1")


def test_ascend_single_op_compile_failed():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        raise ValueError("Single op compile failed")

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "vm_id_9")


def test_version_not_match():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        raise ValueError("""errNo[0x0000000005010001]
                            hcom_ops_kernel_info_store.cc""")

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "vm_id_12")


def test_broadcast_op_distribute_failed():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        raise ValueError("mindspore\ccsrc\plugin\device\ascend\hal\device\ge_runtime\task\hccl_task.cc]"
                         " davinci_model: load task fail, return ret: 1343225860")

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "vm_id_13")


def test_stream_exceeds():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        raise ValueError("AssignAllNodesStream] Total stream number xxx exceeds the limit of 1024, "
                         "secrch details information in mindspore's FAQ.")

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "vm_id_14")


def test_response_is_empty():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        raise ValueError("""
        Traceback (most recent call last):
            ...
        RuntimeError: mindspore/ccsrc/backend/common/session/kernel_build_client.h:107 Response] Response is empty
        """)

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "vm_id_15")


def test_memasycpy_failed():
    # 'Key Log Information' is not in Traceback information
    @ts.proposal(write_file_path="/tmp/")
    def main():
        raise ValueError("Memory async copy failed, device_id=0, stream_id=102, task_id=171, flip_num=0, "
                         "copy_type=10, memcpy_type=0, copy_data_type=0, length=896[FUNC:GetError]"
                         "[FILE:stream.cc][LINE:741]")

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "vm_id_16")


def data_format_not_match():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        raise ValueError("[mindspore/ccsrc/backend/optimizer/ascend/format_type/check_consistency.cc:64] "
                         "CheckFormatForConsistency] Found inconsistent format! input format 0: FracZ, "
                         "selected input format: DefaultFormat")

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "vm_id_17  ")
