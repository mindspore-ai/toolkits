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
    WrapFunctionalOps = yaml.safe_load(f).get('ops')

OpsFunc = {}
for f in dir(ms.ops):
    OpsFunc[f] = getattr(ms.ops, f)


def get_functional_ops():
    global WrapFunctionalOps
    _all_functional_ops = dir(ms.ops)
    return set(WrapFunctionalOps) & set(_all_functional_ops)


class HOOKFunctionalOP(object):
    pass


class FunctionalOPTemplate(HOOKCell):
    def __init__(self, op_name, hook):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Functional_" + str(op_name) + "_"
        super().__init__(hook)

    def construct(self, *args, **kwargs):
        if self.op_name_.startswith('dropout'):
            return args[0] if args else kwargs.get('input')
        return OpsFunc[self.op_name_](*args, **kwargs)


def wrap_functional_op(op_name, hook):
    def functional_op_template(*args, **kwargs):
        return FunctionalOPTemplate(op_name, hook)(*args, **kwargs)

    return functional_op_template


def wrap_functional_ops_and_bind(hook):
    _functional_ops = get_functional_ops()
    for op_name in _functional_ops:
        if callable(OpsFunc[op_name]):
            setattr(HOOKFunctionalOP, "wrap_" + op_name, wrap_functional_op(op_name, hook))
