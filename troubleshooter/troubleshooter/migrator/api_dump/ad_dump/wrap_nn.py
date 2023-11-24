from mindtorch.torch import nn
from . import hook_module as _module
import os
import yaml
cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
with open(yaml_path, 'r') as f:
    WrapNNModule = yaml.safe_load(f).get('Module')

NNCell = {}
for f in dir(nn):
    NNCell[f] = getattr(nn, f)


def get_nn_module():
    global WrapNNModule
    _all_nn_module = dir(nn)
    return set(WrapNNModule) & set(_all_nn_module)
    # return set(_all_nn_module)


def call_decorator(cls, name):
    original_call = cls.__call__
    cls.hook_name = 'wrap_' + name

    def new_call(self, *args, **kwargs):
        if not _module.g_stop_hook:
            _module.g_stop_hook = True
            try:
                result = original_call(self, *args, **kwargs)
            except Exception as e:
                raise e
            finally:
                _module.g_stop_hook = False
        else:
            result = original_call(self, *args, **kwargs)
        return result

    cls.__call__ = new_call
    return cls


def remove_dropout_randomness(cls):

    def new_construct(self, x):
        return x

    cls.construct = new_construct
    return cls


def wrap_nn_module_and_bind():
    _nn_module = get_nn_module()
    for name in _nn_module:
        if name.startswith('Dropout'):
            remove_dropout_randomness(NNCell[name])
        call_decorator(NNCell[name], name)
