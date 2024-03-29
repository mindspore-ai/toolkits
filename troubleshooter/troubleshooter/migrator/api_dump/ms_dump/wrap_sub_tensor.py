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

import mindspore as ms
import os
import yaml

from .hook_cell import HOOKCell

cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
with open(yaml_path, 'r') as f:
    WrapTensorOps = yaml.safe_load(f).get('tensor')

SubTensorFunc = {}
for f in dir(ms.common._stub_tensor.StubTensor):
    SubTensorFunc[f] = getattr(ms.common._stub_tensor.StubTensor, f)


def get_sub_tensor_ops():
    global WrapTensorOps
    _sub_tensor_ops = dir(ms.common._stub_tensor.StubTensor)
    return set(WrapTensorOps) & set(_sub_tensor_ops)


class HOOKSubTensor(object):
    pass


class SubTensorOPTemplate(HOOKCell):

    def __init__(self, op_name, hook):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Tensor_" + str(op_name) + "_"
        super().__init__(hook)

    def construct(self, *args, **kwargs):
        return SubTensorFunc[str(self.op_name_)](*args, **kwargs)


def wrap_sub_tensor_op(op_name, hook):
    def tensor_op_template(*args, **kwargs):
        return SubTensorOPTemplate(op_name, hook)(*args, **kwargs)

    return tensor_op_template


def wrap_sub_tensor_ops_and_bind(hook):
    _tensor_ops = get_sub_tensor_ops()
    for op_name in _tensor_ops:
        if callable(SubTensorFunc[op_name]):
            setattr(HOOKSubTensor, "wrap_" + str(op_name), wrap_sub_tensor_op(op_name, hook))
