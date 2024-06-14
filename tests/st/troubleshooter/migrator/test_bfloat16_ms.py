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
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

import mindspore as ms
from mindspore import Tensor, context, ops, nn
from troubleshooter.migrator import api_dump_init, api_dump_start, api_dump_stop
from tests.st.troubleshooter.migrator.dump.utils import get_csv_npy_stack_list


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        pass

    def construct(self, x):
        pass


def conv2d(x, weight):
    return ops.conv2d(x, weight)


def conv2d_forward_func(x, weight):
    return conv2d(x, weight)


def conv2d_backward_func(x, weight):
    return ops.grad(conv2d_forward_func, (0, 1))(x, weight)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.platform_x86_ascend_training
def test_conv2d_bfloat16_all():
    """
    Feature: api dump support bfloat16.
    Description: api dump collects tensor data for conv2d.
    Expectation: collect tensor data, stack, and statistics for conv2d.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    dump_path = Path(tempfile.mkdtemp(prefix="ms_api_dump_bfloat16"))
    try:
        net = Net()
        api_dump_init(net, dump_path, retain_backward=True)
        api_dump_start(statistic_category = ['max', 'min', 'avg', 'md5', 'l2norm'])
        x = Tensor(np.ones([10, 32, 32, 32]), ms.bfloat16)
        weight = Tensor(np.ones([32, 32, 3, 3]), ms.bfloat16)
        grads = conv2d_backward_func(x, weight)
        dx, dw = grads
        api_dump_stop()
        csv_list, npy_list, stack_list = get_csv_npy_stack_list(
            dump_path, 'mindspore')
        assert 'Functional_conv2d_0_forward_input.0' in npy_list
        assert 'Functional_conv2d_0_forward_input.1' in npy_list
        assert 'Functional_conv2d_0_forward_output' in npy_list
        assert 'Functional_conv2d_0_backward_input' in npy_list
        assert 'Functional_conv2d_0_forward_stack_info' in stack_list
    finally:
        shutil.rmtree(dump_path)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.platform_x86_ascend_training
def test_conv2d_bfloat16_npy():
    """
    Feature: api dump support bfloat16.
    Description: api dump collects tensor data for conv2d.
    Expectation: collect tensor data, stack, and statistics for conv2d.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    dump_path = Path(tempfile.mkdtemp(prefix="ms_api_dump_bfloat16"))
    try:
        net = Net()
        api_dump_init(net, dump_path, retain_backward=True)
        api_dump_start(dump_type = 'npy',statistic_category = ['max', 'min', 'avg', 'md5', 'l2norm'])
        x = Tensor(np.ones([10, 32, 32, 32]), ms.bfloat16)
        weight = Tensor(np.ones([32, 32, 3, 3]), ms.bfloat16)
        grads = conv2d_backward_func(x, weight)
        dx, dw = grads
        api_dump_stop()
        csv_list, npy_list, stack_list = get_csv_npy_stack_list(
            dump_path, 'mindspore')
        assert 'Functional_conv2d_0_forward_input.0' in npy_list
        assert 'Functional_conv2d_0_forward_input.1' in npy_list
        assert 'Functional_conv2d_0_forward_output' in npy_list
        assert 'Functional_conv2d_0_backward_input' in npy_list
        assert 'Functional_conv2d_0_forward_stack_info' in stack_list
    finally:
        shutil.rmtree(dump_path)