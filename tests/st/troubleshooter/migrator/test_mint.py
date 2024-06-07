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
from pathlib import Path
import tempfile
import shutil
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops
from troubleshooter.migrator import api_dump_init, api_dump_start, api_dump_stop
from tests.st.troubleshooter.migrator.dump.utils import get_csv_npy_stack_list


class MaxNet(nn.Cell):
    def construct(self, x):
        return ms.mint.max(x, 0, True)


class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.net = ms.mint.nn.Linear(3, 4)

    def construct(self, x):
        return self.net(x)


def avg_pool2d_forward_func(image, kernel_size, stride=None, padding=0,
                            ceil_mode=False, count_include_pad=True, divisor_override=None,):
    return ms.mint.nn.functional.avg_pool2d(image, kernel_size, stride, padding,
                                            ceil_mode, count_include_pad, divisor_override,)


def avg_pool2d_backward_func(image, kernel_size, stride=None, padding=0, ceil_mode=False,
                             count_include_pad=True, divisor_override=None):
    return ops.grad(avg_pool2d_forward_func, (0,))(image, kernel_size, stride, padding,
                                                   ceil_mode, count_include_pad, divisor_override)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_mint():
    if not "mint" in dir(ms):
        return

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net1 = MaxNet()
    net2 = LinearNet()

    dump_path = Path(tempfile.mkdtemp(prefix="ms_api_dump_mint"))
    try:
        api_dump_init(net2, dump_path, retain_backward=True)
        api_dump_start(statistic_category=['max', 'min', 'avg', 'md5', 'l2norm'])

        input0 = Tensor(np.array([[0.0, 0.3, 0.4, 0.5, 0.1], [3.2, 0.4, 0.1, 2.9, 4.0]]), ms.float32)
        input1 = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), ms.float32)
        output0, _ = net1(input0)
        output1 = net2(input1)
        image = Tensor(np.array([[[4.1702e-1, 7.2032e-1, 1.1437e-4, 3.0223e-1],
                                  [1.4676e-1, 9.2339e-2, 1.8626e-1, 3.4556e-1],
                                  [3.9677e-1, 5.3882e-1, 4.1919e-1, 6.8522e-1],
                                  [2.0445e-1, 8.7812e-1, 2.7338e-2, 6.7047e-1]]]).astype(np.float32))
        out = avg_pool2d_forward_func(image, 2, None, 1, False, True)
        grad = avg_pool2d_backward_func(image, 2, 2, 0, True, False)

        api_dump_stop()
        csv_list, npy_list, stack_list = get_csv_npy_stack_list(dump_path, 'mindspore')
        assert 'Functional_max_0_forward_input.0' in npy_list
        assert 'Functional_max_0_forward_output.0' in npy_list
        assert 'NN_Linear_0_forward_input.0' in npy_list
        assert 'NN_Linear_0_forward_output' in npy_list
        assert 'Functional_avg_pool2d_0_forward_input.0' in npy_list
        assert 'Functional_avg_pool2d_0_forward_output' in npy_list
        assert 'Functional_avg_pool2d_1_forward_input.0' in npy_list
        assert 'Functional_avg_pool2d_1_forward_output' in npy_list
        assert 'Functional_avg_pool2d_1_backward_input' in npy_list
        assert 'Functional_max_0_forward_stack_info' in stack_list
    finally:
        shutil.rmtree(dump_path)
