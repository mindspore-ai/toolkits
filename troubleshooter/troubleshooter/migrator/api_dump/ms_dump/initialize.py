import functools
import os
import builtins

import mindspore as ms

from . import hook_cell
from . import hooks
from . import wrap_functional, wrap_nn, wrap_sub_tensor, wrap_tensor
from .utils import (CompareException, Const, check_file_or_directory_path, print_error_log)
from troubleshooter import log as logger
from mindspore.common.api import _MindsporeFunctionExecutor
try:
    from mindspore.communication import comm_func
    comm_func_label = True
except ImportError:
    comm_func_label = False 

key_wrap = "wrap_"
Key_ops = "wrap_ops."
key_mint_ops = "wrap_mint.ops."
key_mint_nn_functional = "wrap_mint.nn.functional."
key_communication_comm_func = "wrap_communication.comm_func."

def hook_apis():
    for attr_name in dir(wrap_tensor.HOOKTensor):
        if attr_name.startswith(key_wrap):
            setattr(ms.Tensor, attr_name[5:], getattr(wrap_tensor.HOOKTensor, attr_name))
            setattr(ms.common._stub_tensor.StubTensor, attr_name[5:], getattr(wrap_sub_tensor.HOOKSubTensor, attr_name))

    for attr_name in dir(wrap_functional.HOOKFunctionalOP):
        if attr_name.startswith(key_wrap):
            if attr_name.startswith(Key_ops):
                setattr(ms.ops, attr_name[len(Key_ops):], 
                        getattr(wrap_functional.HOOKFunctionalOP, attr_name))
            if attr_name.startswith(key_mint_ops):
                setattr(ms.mint, attr_name[len(key_mint_ops):], 
                        getattr(wrap_functional.HOOKFunctionalOP,attr_name))
            if attr_name.startswith(key_mint_nn_functional):
                setattr(ms.mint.nn.functional, attr_name[len(key_mint_nn_functional):],
                        getattr(wrap_functional.HOOKFunctionalOP, attr_name))
            if comm_func_label and attr_name.startswith(key_communication_comm_func):
                setattr(ms.communication.comm_func, attr_name[len(key_communication_comm_func):],
                        getattr(wrap_functional.HOOKFunctionalOP, attr_name))
    

def restore_apis():
    for attr_name in dir(wrap_tensor.HOOKTensor):
        if attr_name.startswith(key_wrap):
            setattr(ms.Tensor, attr_name[5:], wrap_tensor.TensorFunc[attr_name[5:]])
            setattr(ms.common._stub_tensor.StubTensor, attr_name[5:], wrap_sub_tensor.SubTensorFunc[attr_name[5:]])
    
    for attr_name in dir(wrap_functional.HOOKFunctionalOP):
        if attr_name.startswith(key_wrap):
            if attr_name.startswith(Key_ops):
                setattr(ms.ops, attr_name[len(Key_ops):], 
                        wrap_functional.OpsFunc[wrap_functional.ops_label + attr_name[len(Key_ops):]])
            if attr_name.startswith(key_mint_ops):
                setattr(ms.mint, attr_name[len(key_mint_ops):], 
                        wrap_functional.OpsFunc[wrap_functional.mint_ops_label + attr_name[len(key_mint_ops):]])
            if attr_name.startswith(key_mint_nn_functional):
                setattr(ms.mint.nn.functional, attr_name[len(key_mint_nn_functional):], 
                        wrap_functional.OpsFunc[wrap_functional.mint_nn_func_label + attr_name[len(key_mint_nn_functional):]])
            if comm_func_label and attr_name.startswith(key_communication_comm_func):
                setattr(ms.communication.comm_func, attr_name[len(key_communication_comm_func):], 
                        wrap_functional.OpsFunc[wrap_functional.communication_comm_func_label + attr_name[len(key_communication_comm_func):]])


class MyMindsporeFunctionExecutor(_MindsporeFunctionExecutor):

    def __call__(self, *args, **kwargs):

        restore_apis()
        out = super().__call__(*args, **kwargs)
        hook_apis()

        return out


def initialize_hook(hook):
    wrap_tensor.wrap_tensor_ops_and_bind(hook)
    wrap_sub_tensor.wrap_sub_tensor_ops_and_bind(hook)
    wrap_functional.wrap_functional_ops_and_bind(hook)
    wrap_nn.wrap_nn_cell_and_bind()
    wrap_nn.wrap_optimizer()
    builtins.print = wrap_nn.stop_hook_print

    hook_apis()

    ms.common.api._MindsporeFunctionExecutor = MyMindsporeFunctionExecutor


def register_hook(net, hook, **kwargs):
    dump_mode, dump_config_file = init_dump_config(kwargs)
    pid = os.getpid()
    rank = kwargs.get('rank')
    if rank is None:
        rank = 0
    hooks.make_dump_dirs(rank)

    hook_name = hook.__name__
    logger.info("Start mounting the {} hook function to the model.".format(hook_name))
    hook = functools.partial(hook, pid=pid, dump_mode=dump_mode, dump_config=dump_config_file)
    hook_cell.cell_count.clear()
    hooks.NNCount.clear()
    hooks.range_begin_flag = False
    hooks.range_end_flag = False
    hooks.backward_threading_id = 0

    initialize_hook(hook)
    for _, cell in net.cells_and_names():
        if hasattr(cell, 'hook_name'):
            prefix_nn_name_ = "NN_" + str(cell.hook_name[5:]) + "_"
            cell.register_forward_hook(hook(prefix_nn_name_ + "{}_" + "forward"))
    logger.info("The {} hook function is successfully mounted to the model.".format(hook_name))


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
