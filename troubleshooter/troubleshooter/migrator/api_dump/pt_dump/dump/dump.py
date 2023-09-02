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
import inspect
import json
import math
import os
import stat
import threading
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch

from ... import universal_interface

try:
    import torch_npu
except ImportError:
    is_gpu = True
else:
    is_gpu = False

from ..common.utils import Const, print_info_log, print_warn_log
from ..dump.utils import check_in_api_list, remove_dump_file
from .utils import DumpUtil, make_dump_data_dir

forward_init_status = False
backward_init_status = False

backward_threading_id = 0

NNCount = defaultdict(int)


class DataInfo(object):
    def __init__(self, data, save_data, summary_data, dtype, shape):
        self.data = data
        self.save_data = save_data
        self.summary_data = summary_data
        self.dtype = dtype
        self.shape = shape


def get_not_float_tensor_info(data, compute_summary):
    saved_tensor = data.contiguous().cpu().detach().numpy()
    if compute_summary:
        if data.numel() == 0 or data.dtype == torch.bool:
            tensor_max = math.nan
            tensor_min = math.nan
            tensor_mean = math.nan
        elif len(data.shape) == 0:
            tensor_max = data.cpu().detach().float().numpy().tolist()
            tensor_min = data.cpu().detach().float().numpy().tolist()
            tensor_mean = data.cpu().detach().float().numpy().tolist()
        else:
            tensor_max = torch._C._VariableFunctionsClass.max(data).cpu().detach().float().numpy().tolist()
            tensor_min = torch._C._VariableFunctionsClass.min(data).cpu().detach().float().numpy().tolist()
            tensor_mean = torch._C._VariableFunctionsClass.mean(data.float()).cpu().detach().float().numpy().tolist()
    else:
        tensor_max = math.nan
        tensor_min = math.nan
        tensor_mean = math.nan
    summary_data = [tensor_max, tensor_min, tensor_mean]
    return DataInfo(data, saved_tensor, summary_data, str(data.dtype), tuple(data.shape))


def get_scalar_data_info(data, compute_summary):
    if compute_summary:
        summary_data = [data, data, data]
    else:
        summary_data = [math.nan] * 3
    return DataInfo(data, data, summary_data, str(type(data)), str([]))


def get_float_tensor_info(data, compute_summary):
    saved_tensor = data.contiguous().cpu().detach().numpy()
    if compute_summary:
        tensor_max = torch._C._VariableFunctionsClass.max(data).cpu().detach().float().numpy().tolist()
        tensor_min = torch._C._VariableFunctionsClass.min(data).cpu().detach().float().numpy().tolist()
        tensor_mean = torch._C._VariableFunctionsClass.mean(data).cpu().detach().float().numpy().tolist()
    else:
        tensor_max = math.nan
        tensor_min = math.nan
        tensor_mean = math.nan
    summary_data = [tensor_max, tensor_min, tensor_mean]
    return DataInfo(data, saved_tensor, summary_data, str(data.dtype), tuple(data.shape))


def json_dump_condition(prefix):
    cur_threading_id = threading.current_thread().ident
    global backward_threading_id
    if not backward_threading_id and Const.BACKWARD in prefix:
        backward_threading_id = cur_threading_id
    return (Const.BACKWARD in prefix and backward_threading_id == cur_threading_id) or 'forward' in prefix


def dump_tensor(x, prefix, dump_step, dump_file_name, dump_type):
    global data_info
    if isinstance(x, (tuple, list)) and x:
        for i, item in enumerate(x):
            dump_tensor(item, "{}.{}".format(prefix, i), dump_step, dump_file_name, dump_type)
        return
    elif isinstance(x, torch.Tensor):
        compute_summary = True if dump_type in ['all', 'statistics'] else False
        dump_npy = True if dump_type in ['all', 'npy'] else False
        def backward_hook(grad, get_info):
            nonlocal dump_file_name, dump_step, prefix, dump_npy, compute_summary
            prefix = prefix.replace('_forward_output', '_backward_input')
            data_info_ = get_info(grad, compute_summary)
            dump_data(dump_file_name, dump_step, prefix, data_info_, dump_npy)

        if x.numel() == 0 or len(x.shape) == 0 or not x.is_floating_point():
            if DumpUtil.dump_filter_switch == Const.OFF:
                data_info = get_not_float_tensor_info(x, compute_summary)
                dump_data(dump_file_name, dump_step, prefix, data_info, dump_npy)
                if universal_interface.g_retain_backward and x.requires_grad is True and "_output" in prefix:
                    x.register_hook(functools.partial(backward_hook, get_info=get_not_float_tensor_info))
            else:
                return
        else:
            data_info = get_float_tensor_info(x, compute_summary)
            dump_data(dump_file_name, dump_step, prefix, data_info, dump_npy)
            if universal_interface.g_retain_backward and x.requires_grad is True and "_output" in prefix:
                x.register_hook(functools.partial(backward_hook, get_info=get_float_tensor_info))

    elif DumpUtil.dump_filter_switch == Const.OFF:
        if isinstance(x, bool) or isinstance(x, int) or isinstance(x, float):
            data_info = get_scalar_data_info(x, compute_summary)
            dump_data(dump_file_name, dump_step, prefix, data_info, dump_npy)


def dump_data(dump_file_name, dump_step, prefix, data_info, dump_npy):
    with os.fdopen(os.open(dump_file_name, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR),
                   "a") as f:
        if json_dump_condition(prefix):
            if dump_npy:
                output_path = os.path.join(DumpUtil.dump_data_dir, f'{prefix}.npy')
                np.save(output_path, data_info.save_data)
                os.chmod(output_path, 0x600)
            json.dump([prefix, dump_step, [], data_info.dtype, data_info.shape, data_info.summary_data], f)
            f.write('\n')


@lru_cache()
def is_not_blacklisted(stack_path):
    black_lists = ['/torch/autograd/', '/torch/backends/', '/torch/cuda/',
                   '/torch/distributed/', '/torch/distributions/', '/torch/fft/',
                   '/torch/fx/', '/torch/jit/', '/torch/linalg/', '/torch/nn/',
                   '/torch/onnx/', '/torch/optim/', '/torch/profiler/', '/torch/quantization/',
                   '/troubleshooter/migrator/api_dump/']
    for black in black_lists:
        if black in stack_path:
            return False
    return True


def dump_stack_info(name_template, dump_file, filter_stack):
    stack_str = []
    prefix = name_template.format("stack_info")
    for (_, path, line, func, code, _) in inspect.stack()[3:]:
        if code:
            stack_line = " ".join([
                "File", ", ".join([path, " ".join(["line", str(line)]), " ".join(["in", func]),
                                   " ".join(["\n", code[0].strip() if code else code])])])
        else:
            stack_line = " ".join([
                "File", ", ".join([path, " ".join(["line", str(line)]), " ".join(["in", func]),
                                   " ".join(["\n", code])])])
        if not filter_stack or (filter_stack and is_not_blacklisted(path)):
            stack_str.append(stack_line)

    DumpUtil.dump_stack_dic[prefix] = stack_str
    json_str = json.dumps(DumpUtil.dump_stack_dic, indent=4)

    with os.fdopen(os.open(dump_file, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), "w") as f:
        if DumpUtil.dump_switch_mode in Const.DUMP_MODE:
            if json_dump_condition(prefix):
                f.write(json_str)
        else:
            f.write(json_str)


def dump_api_tensor(dump_step, in_feat, name_template, out_feat, dump_file, dump_type):
    dump_tensor(in_feat, name_template.format("input"), dump_step, dump_file, dump_type)
    dump_tensor(out_feat, name_template.format("output"), dump_step, dump_file, dump_type)


def dump_acc_cmp(name, in_feat, out_feat, dump_step, module):
    dump_path, dump_file_name, dump_stack_file = DumpUtil.get_dump_path()

    if DumpUtil.get_dump_switch():
        if DumpUtil.dump_init_enable:
            DumpUtil.dump_init_enable = False
            DumpUtil.dump_data_dir = make_dump_data_dir(dump_path)
            remove_dump_file(dump_file_name)
            remove_dump_file(dump_stack_file)

        name_prefix = name
        name_template = f"{name_prefix}" + "_{}"
        if DumpUtil.dump_switch_mode == Const.ALL:
            dump_api_tensor(dump_step, in_feat, name_template, out_feat, dump_file_name, DumpUtil.dump_type)
            dump_stack_info(name_template, dump_stack_file, DumpUtil.dump_filter_stack)
        elif DumpUtil.dump_switch_mode == Const.API_LIST:
            if not check_in_api_list(name):
                return
            dump_api_tensor(dump_step, in_feat, name_template, out_feat, dump_file_name, DumpUtil.dump_type)
            dump_stack_info(name_template, dump_stack_file, DumpUtil.dump_filter_stack)
        elif DumpUtil.dump_switch_mode in [Const.RANGE, Const.LIST]:
            if DumpUtil.check_switch_scope(name_prefix):
                dump_api_tensor(dump_step, in_feat, name_template, out_feat, dump_file_name, DumpUtil.dump_type)
                dump_stack_info(name_template, dump_stack_file, DumpUtil.dump_filter_stack)
        else:
            msg = f"Current mode '{DumpUtil.dump_switch_mode}' is not supported. Please use the field in {Const.DUMP_MODE}"
            raise ValueError(msg)


def acl_dump(module, module_name, name_prefix):
    if name_prefix in DumpUtil.backward_input:
        dump_mode_backward_acl_dump(module, module_name, DumpUtil.backward_input.get(name_prefix))
    else:
        forward_acl_dump(module, module_name)


def Op_Need_Trigger(module_name):
    if 'Tensor___getitem___' in module_name:
        return True
    return False


def forward_acl_dump(module, module_name):
    global forward_init_status
    global backward_init_status
    if not forward_init_status and not backward_init_status:
        forward_init_status = True
        torch_npu.npu.init_dump()
        torch_npu.npu.set_dump(DumpUtil.dump_config)
        torch_npu.npu.synchronize()
        if Op_Need_Trigger(module_name):
            module.forward(*module.input_args, **module.input_kwargs).cpu()
        else:
            module.forward(*module.input_args, **module.input_kwargs)
        torch_npu.npu.synchronize()
        torch_npu.npu.finalize_dump()
    del module.input_args
    del module.input_kwargs
    forward_init_status = False
    print_info_log("Dump %s op file." % module_name)


def acl_backward_dump_status(output, grad, module_name):
    if isinstance(output, torch.Tensor):
        output.backward(grad, retain_graph=True)
        return True

    if "_sort_" in module_name :
        output[0].backward(grad, retain_graph=True)
        return True
    return False


def dump_mode_backward_acl_dump(module, module_name, grad_path):
    global forward_init_status
    global backward_init_status
    module_name = module_name.replace(Const.FORWARD, Const.BACKWARD)
    if not forward_init_status and not backward_init_status:
        forward_init_status = True
        module.input_args = list(module.input_args)
        for i, data in enumerate(module.input_args):
            if isinstance(data, torch.Tensor) and data.grad_fn:
                module.input_args[i] = data.detach().requires_grad_()
        output = module.forward(*module.input_args, **module.input_kwargs)
        grad = torch.tensor(np.load(grad_path)).to("npu").requires_grad_()
        torch_npu.npu.init_dump()
        torch_npu.npu.set_dump(DumpUtil.dump_config)
        torch_npu.npu.synchronize()
        if not acl_backward_dump_status(output, grad, module_name):
            print_warn_log("The output of {} is not of tensor type and cannot be automatically derived. "
                            "you can manually construct a single API backward case for ACL dump.".format(module_name))
        torch_npu.npu.synchronize()
        torch_npu.npu.finalize_dump()
    del module.input_args
    del module.input_kwargs
    forward_init_status = False
    print_info_log("Dump %s op file." % module_name)


def acc_cmp_dump(name, **kwargs):
    dump_step = kwargs.get('dump_step', 1)
    pid = kwargs.get('pid')
    DumpUtil.set_dump_config(kwargs.get('dump_config'))
    name_template = name
    if not pid:
        return RuntimeError("Not get the specified process pid.")

    def acc_cmp_hook(module, in_feat, out_feat):
        nonlocal name, name_template
        global NNCount
        if "{}_" in name_template:
            nn_name = name_template.split("_")[1]
            id = NNCount[nn_name]
            NNCount[nn_name] = id + 1
            name = name_template.format(id)
        if pid == os.getpid():
            dump_acc_cmp(name, in_feat, out_feat, dump_step, module)
        if hasattr(module, "input_args"):
            del module.input_args
        if hasattr(module, "input_kwargs"):
            del module.input_kwargs

    return acc_cmp_hook
