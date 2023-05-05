import functools
import mindspore as ms
import os

import wrap_functional
import wrap_tensor
from hooks import make_dump_dirs
from utils import print_error_log, Const, CompareException, check_file_or_directory_path, print_info_log


def initialize_hook(hook):
    wrap_tensor.wrap_tensor_ops_and_bind(hook)
    for attr_name in dir(wrap_tensor.HOOKTensor):
        if attr_name.startswith("wrap_"):
            setattr(ms.Tensor, attr_name[5:], getattr(wrap_tensor.HOOKTensor, attr_name))

    wrap_functional.wrap_functional_ops_and_bind(hook)
    for attr_name in dir(wrap_functional.HOOKFunctionalOP):
        if attr_name.startswith("wrap_"):
            setattr(ms.ops, attr_name[5:], getattr(wrap_functional.HOOKFunctionalOP, attr_name))


def register_hook(net, hook, **kwargs):
    dump_mode, dump_config_file = init_dump_config(kwargs)
    pid = os.getpid()
    rank = kwargs.get('rank')
    if rank is None:
        rank = 0
    make_dump_dirs(rank, pid)
    hook_name = hook.__name__

    print_info_log("Start mounting the {} hook function to the model.".format(hook_name))
    hook = functools.partial(hook, pid=pid, dump_mode=dump_mode, dump_config=dump_config_file)
    print_info_log("The {} hook function is successfully mounted to the model.".format(hook_name))

    initialize_hook(hook)
    for name, cell in net.name_cells().items():
        if hasattr(cell, 'hook_name'):
            prefix_nn_name_ = "NN_" + str(cell.hook_name[5:]) + "_"
            cell.register_forward_hook(hook(prefix_nn_name_ + "{}_" + "forward", forward=True))
            cell.register_backward_hook(hook(prefix_nn_name_ + "{}_" + "backward", forward=False))


def init_dump_config(kwargs):
    dump_mode = kwargs.get('dump_mode', "api")
    dump_config = kwargs.get('dump_config')
    dump_config_file = ''
    if dump_mode not in Const.SUPPORT_DUMP_MODE:
        print_error_log("dump_mode only support %s" % Const.SUPPORT_DUMP_MODE)
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    if dump_mode == "acl":
        if dump_config is None:
            print_error_log("dump_mode is acl mode, dump_config must be configured.")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)
        dump_config_file = os.path.realpath(dump_config)
        check_file_or_directory_path(dump_config_file)
        if not dump_config.endswith(".json"):
            print_error_log("dump_config must be configure json file.")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)
    return dump_mode, dump_config_file
