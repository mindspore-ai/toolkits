# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import numpy as np
import pytest
from mindspore.communication import init, get_rank
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import tempfile
import shutil
from mindspore import dtype as mstype
from pathlib import Path
from troubleshooter.migrator import api_dump_init, api_dump_start, api_dump_stop
from tests.st.troubleshooter.migrator.dump.utils import get_pkl_npy_stack_list
try:
    from mindspore.communication import comm_func
    comm_func_label = True
except ImportError:
    comm_func_label = False    


class NetNull(nn.Cell):
    def __init__(self):
        super(NetNull, self).__init__()
        pass

    def construct(self, x):
        pass
    
def all_reduce_dump(x):
    if comm_func_label:
        return  comm_func.all_reduce(x)
    else:
        return

def test_api_dump_communicate():
    init()
    if not comm_func_label:
        return      
    net = NetNull()
    dump_path = Path(tempfile.mkdtemp(prefix="ms_api_dump_communication"))
    try:
        api_dump_init(net, dump_path, retain_backward=True)
        api_dump_start()
        input = ms.Tensor(np.ones([3, 4]).astype(np.float32))
        expect_output = [[2, 2, 2, 2],[2, 2, 2, 2],[2, 2, 2, 2]]
        output = ops.grad(all_reduce_dump)(input)
        api_dump_stop()
        pkl_list, npy_list, stack_list = get_pkl_npy_stack_list(dump_path, 'mindspore')
        assert 'Functional_all_reduce_0_backward_input' in npy_list
        assert 'Functional_all_reduce_0_forward_input.0' in npy_list
        assert 'Functional_all_reduce_0_forward_output' in npy_list 
    finally:
        shutil.rmtree(dump_path) 
