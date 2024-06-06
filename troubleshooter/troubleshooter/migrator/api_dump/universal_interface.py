# Copyright 2022-2023 Huawei Technologies Co., Ltd
#
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
# ============================================================================
from __future__ import absolute_import

import os
import re

from troubleshooter import FRAMEWORK_TYPE
from troubleshooter.common.util import enum_check, type_check
from troubleshooter import log as logger

if "torch" in FRAMEWORK_TYPE:
    import torch
    from torch.optim import lr_scheduler
    from troubleshooter.migrator.api_dump.pt_dump import acc_cmp_dump as pt_acc_cmp_dump
    from troubleshooter.migrator.api_dump.pt_dump import register_hook as pt_register_hook
    from troubleshooter.migrator.api_dump.pt_dump import set_dump_path as pt_set_dump_path
    from troubleshooter.migrator.api_dump.pt_dump import set_dump_switch as pt_set_dump_switch
    from troubleshooter.migrator.api_dump.pt_dump.hook_module.wrap_module import wrap_lr_scheduler

    _LRScheduler = getattr(lr_scheduler, "_LRScheduler", None)
    if _LRScheduler:
        wrap_lr_scheduler(_LRScheduler)

if "mindspore" in FRAMEWORK_TYPE:
    import mindspore
    from troubleshooter.migrator.api_dump.ms_dump import acc_cmp_dump as ms_acc_cmp_dump
    from troubleshooter.migrator.api_dump.ms_dump import register_hook as ms_register_hook
    from troubleshooter.migrator.api_dump.ms_dump import set_dump_path as ms_set_dump_path
    from troubleshooter.migrator.api_dump.ms_dump import set_dump_switch as ms_set_dump_switch

if "mindtorch" in FRAMEWORK_TYPE:
    import mindtorch.torch
    from troubleshooter.migrator.api_dump.ad_dump import acc_cmp_dump as ad_acc_cmp_dump
    from troubleshooter.migrator.api_dump.ad_dump import register_hook as ad_register_hook
    from troubleshooter.migrator.api_dump.ms_dump import set_dump_path as ad_set_dump_path
    from troubleshooter.migrator.api_dump.ms_dump import set_dump_switch as ad_set_dump_switch

API_DUMP_FRAMEWORK_TYPE = None
g_retain_backward = None


def _check_scope_format(scope):
    res = True
    pattern = r'_\d+$'
    for item in scope:
        res &= bool(re.search(pattern, item))
    return res


def check_mode_and_scope(mode, scope):
    support_mode = {'all', 'list', 'api_list', 'range'}
    enum_check(mode, 'mode', support_mode)

    if mode == 'all':
        if scope:
            raise ValueError("For 'api_dump_start',  when mode is 'all', "
                             f"the 'scope' muse be None, but currently it is '{scope}'.")
    elif mode == 'list':
        type_check(scope, 'scope', list)
        if len(scope) == 0:
            raise ValueError("For 'api_dump_start',  current mode is 'list', "
                             "'scope' param set invalid, it's should not be an empty list.")
        if not _check_scope_format(scope):
            raise ValueError("For 'api_dump_start',  current mode is 'list', "
                             "'scope' param set invalid. The list item "
                             "should be in the 'Type_Name_ID' format.")
    elif mode == 'api_list':
        type_check(scope, 'scope', list)
        if len(scope) == 0:
            raise ValueError("For 'api_dump_start',  current mode is 'api_list', "
                             "'scope' param set invalid, it's should not be an empty list.")
    elif mode == 'range':
        type_check(scope, 'scope', list)
        if len(scope) != 2:
            raise ValueError("For 'api_dump_start',  current mode is 'range', "
                             "'scope' param set invalid, it's must be [start, end].")
        if not _check_scope_format(scope):
            raise ValueError("For 'api_dump_start',  current mode is 'range', "
                             "'scope' param set invalid. The interval endpoints "
                             "should be in the 'Type_Name_ID' format.")
    else:
        raise ValueError("For 'api_dump_start',  current mode is invalid. ")


def api_dump_init(net, output_path=os.path.join(os.getcwd(), "ts_api_dump"), *, retain_backward=False,
                  **kwargs):
    global API_DUMP_FRAMEWORK_TYPE
    global g_retain_backward
    g_retain_backward = retain_backward
    type_check(retain_backward, 'retain_backward', bool)
    compare_statedict = kwargs.get('compare_statedict', False)

    logger.user_attention("For precision comparison, the probability p in the dropout method is set to 0.")
    logger.user_attention("Please disable the shuffle function of the dataset "
                          "before running the program.")

    # mindtorch must be before torch, because mindtorch Module will be recognized as torch Module when mstorch_enable().
    if "mindtorch" in FRAMEWORK_TYPE and isinstance(net, mindtorch.torch.nn.Module):
        mindtorch.module_hooker.torch_enable()
        API_DUMP_FRAMEWORK_TYPE = "mindtorch"
        ad_set_dump_path(output_path)
        ad_register_hook(net, ad_acc_cmp_dump, compare_statedict=compare_statedict)
        mindtorch.module_hooker.torch_pop()
    elif "torch" in FRAMEWORK_TYPE and isinstance(net, torch.nn.Module):
        API_DUMP_FRAMEWORK_TYPE = "torch"
        pt_set_dump_path(output_path)
        pt_register_hook(net, pt_acc_cmp_dump, compare_statedict=compare_statedict, rank=os.getenv("RANK"))
    elif "mindspore" in FRAMEWORK_TYPE and isinstance(net, mindspore.nn.Cell):
        API_DUMP_FRAMEWORK_TYPE = "mindspore"
        ms_set_dump_path(output_path)
        ms_register_hook(net, ms_acc_cmp_dump, rank=os.getenv("RANK_ID"))
    else:
        raise TypeError("For 'troubleshooter.api_dump.init' function, the type of argument 'net' must be "
                        f"mindspore.nn.Cell, torch.nn.Module or mindtorch.torch.nn.Module, but got {type(net)}.")


def api_dump_start(mode = 'all', scope = None, dump_type = "all", filter_data = True, filter_stack = True, overflow_check = False, statistic_category = ['max', 'min', 'l2norm']):
    check_mode_and_scope(mode, scope)
    if scope is None:
        scope = []
    support_dump_type = {'all', 'statistics', 'stack', 'npy'}
    support_statistic_category = {'min', 'avg', 'max', 'md5','l2norm'}
    enum_check(dump_type, 'support_dump_type', support_dump_type)
    if statistic_category is not None:
        for param in statistic_category:
            enum_check(param, 'statistic_category', support_statistic_category)
    type_check(filter_data, 'filter_data', bool)
    filter_switch = 'ON' if filter_data else 'OFF'
    if API_DUMP_FRAMEWORK_TYPE == "torch":
        pt_set_dump_switch("ON", mode, scope=scope, api_list=scope, filter_switch=filter_switch,
                           dump_type=dump_type, filter_stack=filter_stack, statistic_category=statistic_category)
    elif API_DUMP_FRAMEWORK_TYPE == "mindspore":
        ms_set_dump_switch("ON", mode, scope=scope, api_list=scope, filter_switch=filter_switch,
                           dump_type=dump_type, filter_stack=filter_stack, overflow=overflow_check, statistic_category=statistic_category)
    elif API_DUMP_FRAMEWORK_TYPE == "mindtorch":
        mindtorch.module_hooker.torch_enable()
        ad_set_dump_switch("ON", mode, scope=scope, api_list=scope, filter_switch=filter_switch,
                           dump_type=dump_type, filter_stack=filter_stack, statistic_category=statistic_category)
        mindtorch.module_hooker.torch_pop()
    else:
        raise RuntimeError("You must call 'troubleshooter.api_dump.init' before calling"
                           "'troubleshooter.api_dump.start' function.")


def api_dump_stop():
    if API_DUMP_FRAMEWORK_TYPE == "torch":
        pt_set_dump_switch("OFF")
    elif API_DUMP_FRAMEWORK_TYPE == "mindspore":
        ms_set_dump_switch("OFF")
    elif API_DUMP_FRAMEWORK_TYPE == "mindtorch":
        mindtorch.module_hooker.torch_enable()
        ad_set_dump_switch("OFF")
        mindtorch.module_hooker.torch_pop()
    else:
        raise RuntimeError("You must call 'troubleshooter.api_dump.init' before calling"
                           "'troubleshooter.api_dump.stop' function.")
