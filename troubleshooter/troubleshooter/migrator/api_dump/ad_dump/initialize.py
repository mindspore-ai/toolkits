import functools
import os
import msadapter

from .hooks import make_adapter_dump_dirs
from . import wrap_torch, wrap_nn, wrap_tensor, wrap_functional
from troubleshooter.migrator.api_dump.ms_dump.initialize import init_dump_config
from troubleshooter.migrator.api_dump.ms_dump.utils import print_info_log

def initialize_hook(hook):
    wrap_tensor.wrap_tensor_ops_and_bind(hook)
    for attr_name in dir(wrap_tensor.HOOKTensor):
        if attr_name.startswith("wrap_"):
            setattr(msadapter.pytorch.Tensor, attr_name[5:], getattr(wrap_tensor.HOOKTensor, attr_name))

    wrap_torch.wrap_torch_ops_and_bind(hook)
    for attr_name in dir(wrap_torch.HOOKTorchOps):
        if attr_name.startswith("wrap_"):
            setattr(msadapter.pytorch, attr_name[5:], getattr(wrap_torch.HOOKTorchOps, attr_name))

    wrap_functional.wrap_functional_ops_and_bind(hook)
    for attr_name in dir(wrap_functional.HOOKFunctionalOP):
        if attr_name.startswith("wrap_"):
            setattr(msadapter.pytorch.nn.functional, attr_name[5:], getattr(wrap_functional.HOOKFunctionalOP, attr_name))

    wrap_nn.wrap_nn_module_and_bind()

def register_hook(net, hook, **kwargs):
    dump_mode, dump_config_file = init_dump_config(kwargs)
    pid = os.getpid()
    rank = kwargs.get('rank')
    if rank is None:
        rank = 0
    make_adapter_dump_dirs(rank)

    hook_name = hook.__name__
    print_info_log("Start mounting the {} hook function to the model.".format(hook_name))
    hook = functools.partial(hook, pid=pid, dump_mode=dump_mode, dump_config=dump_config_file)

    initialize_hook(hook)
    for _, module in net.named_modules():
        if hasattr(module, 'hook_name'):
            prefix_nn_name_ = "NN_" + str(module.hook_name[5:]) + "_"
            module.register_forward_hook(hook(prefix_nn_name_ + "{}_" + "forward"))
    print_info_log("The {} hook function is successfully mounted to the model.".format(hook_name))