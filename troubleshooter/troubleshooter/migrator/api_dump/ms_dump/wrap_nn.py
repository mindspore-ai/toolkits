# Copyright 2023 Huawei Technologies Co., Ltd
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
import os

import mindspore as ms
import numpy as np
import yaml
from mindspore import nn

from . import hook_cell as _cell

cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
with open(yaml_path, 'r') as f:
    WrapNNCell = yaml.safe_load(f).get('nn')

NNCell = {}
for f in dir(ms.nn):
    NNCell[f] = getattr(ms.nn, f)


def get_nn_cell():
    global WrapNNCell
    _all_nn_cell = dir(ms.nn)
    return set(WrapNNCell) & set(_all_nn_cell)


def call_decorator(cls, name):
    original_call = cls.__call__
    cls.hook_name = 'wrap_' + name

    def new_call(self, *args, **kwargs):
        if not _cell.g_stop_hook:
            _cell.g_stop_hook = True
            try:
                result = original_call(self, *args, **kwargs)
            except Exception as e:
                raise e
            finally:
                _cell.g_stop_hook = False
        else:
            result = original_call(self, *args, **kwargs)
        return result

    cls.__call__ = new_call
    return cls


def remove_dropout_randomness(cls):

    def new_construct(self, x):
        return x

    cls.construct = new_construct
    return cls


def wrap_nn_cell_and_bind():
    _nn_cell = get_nn_cell()
    for name in _nn_cell:
        if name.startswith('Dropout'):
            remove_dropout_randomness(NNCell[name])
        call_decorator(NNCell[name], name)


if __name__ == '__main__':
    wrap_nn_cell_and_bind()
    net = nn.Conv1d(120, 240, 4, has_bias=False, weight_init='normal')
    x = ms.Tensor(np.ones([1, 120, 640]), ms.float32)
    output = net(x).shape
    print(output)
