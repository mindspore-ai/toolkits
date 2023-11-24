import os

import mindtorch.torch as torch
import yaml

from .hook_module import HOOKModule

cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
with open(yaml_path, 'r') as f:
    WrapFunctionalOps = yaml.safe_load(f).get('functional')


FunctionalFunc = {}
for f in dir(torch.nn.functional):
    FunctionalFunc[f] = getattr(torch.nn.functional, f)


def get_functional_ops():
    global WrapFunctionalOps
    _all_functional_ops = dir(torch.nn.functional)
    return set(WrapFunctionalOps) & set(_all_functional_ops)


class HOOKFunctionalOP(object):
    pass


class FunctionalOPTemplate(HOOKModule):
    def __init__(self, op_name, hook):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Functional_" + str(op_name) + "_"
        super().__init__(hook)

    def forward(self, *args, **kwargs):
        if self.op_name_.startswith("dropout"):
            return args[0] if args else kwargs.get("input")
        return FunctionalFunc[str(self.op_name_)](*args, **kwargs)


def wrap_functional_op(op_name, hook):
    def functional_op_template(*args, **kwargs):
        return FunctionalOPTemplate(op_name, hook)(*args, **kwargs)

    return functional_op_template


def wrap_functional_ops_and_bind(hook):
    _functional_ops = get_functional_ops()
    for op_name in _functional_ops:
        if callable(FunctionalFunc[op_name]):
            setattr(HOOKFunctionalOP, "wrap_" + op_name, wrap_functional_op(op_name, hook))