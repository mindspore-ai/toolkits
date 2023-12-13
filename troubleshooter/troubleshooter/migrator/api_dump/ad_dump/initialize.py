import functools
import os
import mindtorch

from .hooks import make_adapter_dump_dirs, make_pth_dir
from . import hook_module
from . import hooks
from . import wrap_torch, wrap_nn, wrap_tensor, wrap_functional
from troubleshooter.migrator.api_dump.ms_dump.initialize import init_dump_config
from troubleshooter.migrator.api_dump.ms_dump.utils import print_info_log
from troubleshooter import log as logger

def initialize_hook(hook):
    wrap_tensor.wrap_tensor_ops_and_bind(hook)
    for attr_name in dir(wrap_tensor.HOOKTensor):
        if attr_name.startswith("wrap_"):
            setattr(mindtorch.torch.Tensor, attr_name[5:], getattr(wrap_tensor.HOOKTensor, attr_name))

    wrap_torch.wrap_torch_ops_and_bind(hook)
    for attr_name in dir(wrap_torch.HOOKTorchOps):
        if attr_name.startswith("wrap_"):
            setattr(mindtorch.torch, attr_name[5:], getattr(wrap_torch.HOOKTorchOps, attr_name))

    wrap_functional.wrap_functional_ops_and_bind(hook)
    for attr_name in dir(wrap_functional.HOOKFunctionalOP):
        if attr_name.startswith("wrap_"):
            setattr(mindtorch.torch.nn.functional, attr_name[5:], getattr(wrap_functional.HOOKFunctionalOP, attr_name))

    wrap_nn.wrap_nn_module_and_bind()

def register_hook(net, hook, **kwargs):
    dump_mode, dump_config_file = init_dump_config(kwargs)
    pid = os.getpid()
    compare_statedict = kwargs.get('compare_statedict', False)
    rank = kwargs.get('rank')
    if rank is None:
        rank = 0
    make_adapter_dump_dirs(rank)
    if compare_statedict:
        ad_pth_path = make_pth_dir()
        if os.path.exists(ad_pth_path):
            logger.user_attention(f"Found existing state_dict info in '{ad_pth_path}'.")
            os.remove(ad_pth_path)
            logger.user_attention("Existing state_dict info removed.")
        mindtorch.torch.save(net.state_dict(), ad_pth_path)
        os.chmod(ad_pth_path, 0o400)
        logger.user_attention(f"MindTorch model's state_dict has been saved to {ad_pth_path}.")
    hook_name = hook.__name__
    print_info_log("Start mounting the {} hook function to the model.".format(hook_name))
    hook = functools.partial(hook, pid=pid, dump_mode=dump_mode, dump_config=dump_config_file)
    hook_module.module_count.clear()
    hooks.NNCount.clear()
    hooks.range_begin_flag = False
    hooks.range_end_flag = False
    hooks.backward_threading_id = 0

    initialize_hook(hook)
    for m_name, module in net.named_modules():
        m_name = m_name.replace('.', '_')
        if module is not None and m_name is not "":
            module.register_forward_hook(hook(f"LAYER_{m_name}_forward"))
        if hasattr(module, 'hook_name'):
            prefix_nn_name_ = "NN_" + str(module.hook_name[5:]) + "_"
            module.register_forward_hook(hook(prefix_nn_name_ + "{}_" + "forward"))
    print_info_log("The {} hook function is successfully mounted to the model.".format(hook_name))