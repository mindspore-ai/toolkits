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
import mindspore.nn as nn
import mindspore.ops as ops
import tempfile
import shutil
from mindspore import Tensor, ops
from pathlib import Path
from troubleshooter.migrator import api_dump_init, api_dump_start, api_dump_stop
from tests.st.troubleshooter.migrator.dump.utils import get_csv_npy_stack_list
   
class NetNull(nn.Cell):
    def __init__(self):
        super(NetNull, self).__init__()
        pass

    def construct(self, x):
        pass

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dump_complete():    
    net = NetNull()
    dump_path = Path(tempfile.mkdtemp(prefix="ms_dump_complete"))
    try:
        api_dump_init(net, dump_path, retain_backward=True)
        api_dump_start(filter_data = False, statistic_category = ['min', 'avg', 'l2norm'])
        input = Tensor(np.array([[3, 6, 9], [3, 6, 9]]))
        boundaries = list(np.array([1., 3., 5., 7., 9.]))
        output = ops.bucketize(input, boundaries, right=True)
        api_dump_stop()
        csv_list, npy_list, stack_list = get_csv_npy_stack_list(dump_path, 'mindspore')
        assert output.shape == (2, 3)
        assert 'Functional_bucketize_0_forward_input.0' in csv_list
        assert 'Functional_bucketize_0_forward_input.1.0' in csv_list
        assert 'Functional_bucketize_0_forward_output' in csv_list        
        assert 'Functional_bucketize_0_forward_output' in npy_list
        assert 'Functional_bucketize_0_forward_stack_info' in stack_list 
    finally:
        shutil.rmtree(dump_path) 
