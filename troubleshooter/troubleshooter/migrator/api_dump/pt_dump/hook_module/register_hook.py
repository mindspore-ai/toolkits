#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2019-2020. Huawei Technologies Co., Ltd. All rights reserved.
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
"""

import functools
import os

import torch

from ... import universal_interface
from ..common import global_manage
from ..common.utils import (CompareException, Const, check_file_or_directory_path, get_process_rank,
                            print_error_log, print_info_log, print_warn_log)
from ..dump.utils import make_dump_dirs
from . import wrap_functional, wrap_module, wrap_tensor, wrap_torch, wrap_vf

try:
    import torch_npu
except ImportError:
    is_gpu = True
else:
    is_gpu = False
    from . import wrap_npu_custom

make_dir_flag = True

global_manage.init()

def initialize_hook(hook):
    wrap_tensor.wrap_tensor_ops_and_bind(hook)
    for attr_name in dir(wrap_tensor.HOOKTensor):
        if attr_name.startswith("wrap_"):
            setattr(torch.Tensor, attr_name[5:], getattr(wrap_tensor.HOOKTensor, attr_name))

    wrap_torch.wrap_torch_ops_and_bind(hook)
    for attr_name in dir(wrap_torch.HOOKTorchOP):
        if attr_name.startswith("wrap_"):
            setattr(torch, attr_name[5:], getattr(wrap_torch.HOOKTorchOP, attr_name))

    wrap_functional.wrap_functional_ops_and_bind(hook)
    for attr_name in dir(wrap_functional.HOOKFunctionalOP):
        if attr_name.startswith("wrap_"):
            setattr(torch.nn.functional, attr_name[5:], getattr(wrap_functional.HOOKFunctionalOP, attr_name))

    wrap_vf.wrap_vf_ops_and_bind(hook)
    for attr_name in dir(wrap_vf.HOOKVfOP):
        if attr_name.startswith("wrap_"):
            setattr(torch._VF, attr_name[5:], getattr(wrap_vf.HOOKVfOP, attr_name))

    if not is_gpu:
        wrap_npu_custom.wrap_npu_ops_and_bind(hook)
        for attr_name in dir(wrap_npu_custom.HOOKNpuOP):
            if attr_name.startswith("wrap_"):
                setattr(torch_npu, attr_name[5:], getattr(wrap_npu_custom.HOOKNpuOP, attr_name))

    wrap_module.wrap_nn_module_and_bind()


def register_hook(model, hook, **kwargs):
    global make_dir_flag
    assert hasattr(model, "named_modules"), "Please register hooks to nn.Module."
    dump_step = kwargs.get('dump_step', 1)
    overflow_nums = kwargs.get('overflow_nums', 1)
    dump_mode, dump_config_file = init_dump_config(kwargs)

    pid = os.getpid()
    rank = 0
    need_clear = True
    if rank is None:
        rank, need_clear = get_process_rank(model)
    if make_dir_flag:
        make_dump_dirs(rank)
        make_dir_flag = False
    hook_name = hook.__name__

    if "overflow_check" in hook_name and not is_gpu:
        if hasattr(torch_npu._C, "_enable_overflow_npu"):
            torch_npu._C._enable_overflow_npu()
            print_info_log("Enable overflow function success.")            
        else:
            print_warn_log("Api '_enable_overflow_npu' is not exist, "
                           "the overflow detection function on milan platform maybe not work! "
                           "please check the version of software torch_npu.")
        # In NPU scene, clear the overflow flag before overflow detection
        if need_clear:
            torch_npu.npu.set_device(rank)
            torch_npu._C._clear_overflow_npu()

    print_info_log("Start mounting the {} hook function to the model.".format(hook_name))
    hook = functools.partial(hook, dump_step=dump_step, overflow_nums=overflow_nums, pid=pid,
                             dump_mode=dump_mode, dump_config=dump_config_file)

    initialize_hook(hook)

    for _, module in model.named_modules():
        if hasattr(module, 'hook_name'):
            prefix_nn_name_ = "NN_" + str(module.hook_name[5:]) + "_"
            module.register_forward_hook(hook(prefix_nn_name_ + "{}_" + "forward"))
    print_info_log("The {} hook function is successfully mounted to the model.".format(hook_name))


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
