from mindtorch.torch import Tensor
from .hook_module import HOOKModule
import os
import yaml
cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
with open(yaml_path, 'r') as f:
    WrapTensorOps = yaml.safe_load(f).get('tensor')

TensorFunc = {}
for f in dir(Tensor):
    TensorFunc[f] = getattr(Tensor, f)


def get_tensor_ops():
    global WrapTensorOps
    _tensor_ops = dir(Tensor)
    return set(WrapTensorOps) & set(_tensor_ops)


class HOOKTensor(object):
    pass


class TensorOPTemplate(HOOKModule):

    def __init__(self, op_name, hook):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Tensor_" + str(op_name) + "_"
        super().__init__(hook)

    def forward(self, *args, **kwargs):
        return TensorFunc[str(self.op_name_)](*args, **kwargs)


def wrap_tensor_op(op_name, hook):
    def tensor_op_template(*args, **kwargs):
        return TensorOPTemplate(op_name, hook)(*args, **kwargs)

    return tensor_op_template


def wrap_tensor_ops_and_bind(hook):
    _tensor_ops = get_tensor_ops()
    for op_name in _tensor_ops:
        if callable(TensorFunc[op_name]):
            setattr(HOOKTensor, "wrap_" + str(op_name), wrap_tensor_op(op_name, hook))