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


ops_label = "ops."
mint_ops_label = "mint.ops."
mint_nn_func_label = "mint.nn.functional."
cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
WrapFunctionalOps = []
with open(yaml_path, 'r') as f:
    WrapFunctionalOps.extend([ops_label + f for f in yaml.safe_load(f).get('ops')])
with open(yaml_path, 'r') as f:
    WrapFunctionalOps.extend([mint_ops_label + f for f in yaml.safe_load(f).get('mint.ops')])
with open(yaml_path, 'r') as f:
    WrapFunctionalOps.extend([mint_nn_func_label + f for f in yaml.safe_load(f).get('mint.nn.functional')])

OpsFunc = {}
for f in dir(ms.ops):
    OpsFunc[ops_label + f] = getattr(ms.ops, f)
if "mint" in dir(ms):
    for f in dir(ms.mint):
        OpsFunc[mint_ops_label + f] = getattr(ms.mint, f)
    for f in dir(ms.mint.nn.functional):
        OpsFunc[mint_nn_func_label + f] = getattr(ms.mint.nn.functional, f)


def get_functional_ops():
    global WrapFunctionalOps
    _all_functional_ops = []
    _all_functional_ops.extend([ops_label + f for f in dir(ms.ops)])
    if "mint" in dir(ms):
        _all_functional_ops.extend([mint_ops_label + f for f in dir(ms.mint)])
        _all_functional_ops.extend([mint_nn_func_label + f for f in dir(ms.mint.nn.functional)])
    return set(WrapFunctionalOps) & set(_all_functional_ops)


class HOOKFunctionalOP(object):
    pass


class FunctionalOPTemplate(HOOKCell):
    def __init__(self, op_name, hook):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Functional_" + str(op_name.split(".")[-1]) + "_"
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
