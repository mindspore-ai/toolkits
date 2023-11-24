import mindtorch
from .hook_module import HOOKModule
import os
import yaml
# TODO: do we need support list?
cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
with open(yaml_path, 'r') as f:
    WrapFunctionalOps = yaml.safe_load(f).get('torch')

TorchFunc = {}
for f in dir(mindtorch.torch):
    TorchFunc[f] = getattr(mindtorch.torch, f)


def get_torch_ops():
    global WrapFunctionalOps
    _torch_ops = dir(mindtorch.torch)
    return set(WrapFunctionalOps) & set(_torch_ops)
    # return set(_torch_ops)


class HOOKTorchOps(object):
    pass


class TorchOpsTemplate(HOOKModule):

    def __init__(self, op_name, hook):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Torch_" + str(op_name) + "_"
        super().__init__(hook)

    def forward(self, *args, **kwargs):
        return TorchFunc[str(self.op_name_)](*args, **kwargs)


def wrap_torch_ops(op_name, hook):

    def torch_ops_template(*args, **kwargs):
        return TorchOpsTemplate(op_name, hook)(*args, **kwargs)

    return torch_ops_template


def wrap_torch_ops_and_bind(hook):
    _torch_ops = get_torch_ops()
    for op_name in _torch_ops:
        if callable(TorchFunc[op_name]):
            setattr(HOOKTorchOps, "wrap_" + op_name, wrap_torch_ops(op_name, hook))