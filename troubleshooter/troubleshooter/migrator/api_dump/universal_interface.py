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

from troubleshooter import FRAMEWORK_TYPE
from troubleshooter.common.util import enum_check, type_check
from troubleshooter.migrator.api_dump.ms_dump import acc_cmp_dump as ms_acc_cmp_dump
from troubleshooter.migrator.api_dump.ms_dump import register_hook as ms_register_hook
from troubleshooter.migrator.api_dump.ms_dump import set_dump_path as ms_set_dump_path
from troubleshooter.migrator.api_dump.ms_dump import set_dump_switch as ms_set_dump_switch
from troubleshooter.migrator.api_dump.pt_dump import acc_cmp_dump as pt_acc_cmp_dump
from troubleshooter.migrator.api_dump.pt_dump import register_hook as pt_register_hook
from troubleshooter.migrator.api_dump.pt_dump import set_dump_path as pt_set_dump_path
from troubleshooter.migrator.api_dump.pt_dump import set_dump_switch as pt_set_dump_switch
from troubleshooter.migrator.api_dump.pt_dump.common.utils import print_attent_log

if "torch" in FRAMEWORK_TYPE:
    import torch

if "mindspore" in FRAMEWORK_TYPE:
    import mindspore

API_DUMP_FRAMEWORK_TYPE = None
g_retain_backward = None


def api_dump_init(net, output_path=os.path.join(os.getcwd(), "ts_api_dump"), *, retain_backward=False):
    global API_DUMP_FRAMEWORK_TYPE
    global g_retain_backward
    g_retain_backward = retain_backward

    print_attent_log("For precision comparison, the probability p in the dropout method is set to 0.")
    print_attent_log("Please disable the shuffle function of the dataset "
                     "before running the program.")

    if "torch" in FRAMEWORK_TYPE and isinstance(net, torch.nn.Module):
        API_DUMP_FRAMEWORK_TYPE = "torch"
        pt_set_dump_path(output_path)
        pt_register_hook(net, pt_acc_cmp_dump)
    elif "mindspore" in FRAMEWORK_TYPE and isinstance(net, mindspore.nn.Cell):
        API_DUMP_FRAMEWORK_TYPE = "mindspore"
        ms_set_dump_path(output_path)
        ms_register_hook(net, ms_acc_cmp_dump)
    else:
        raise TypeError(f"For 'troubleshooter.api_dump.init' function, the type of argument 'net' must be mindspore.nn.Cell or torch.nn.Module, but got {type(net)}.")


def api_dump_start(mode='all', scope=None, dump_type="all", filter_data=True, filter_stack=True):
    if scope is None:
        scope = []
    support_mode = {'all', 'list', 'api_list', 'range'}
    support_dump_type = {'all', 'statistics', 'stack', 'npy'}
    enum_check(mode, 'mode', support_mode)
    enum_check(dump_type, 'support_dump_type', support_dump_type)
    type_check(filter_data, 'filter_data', bool)
    filter_switch = 'ON' if filter_data else 'OFF'
    if API_DUMP_FRAMEWORK_TYPE == "torch":
        pt_set_dump_switch("ON", mode, scope=scope, api_list=scope, filter_switch=filter_switch,
                           dump_type=dump_type, filter_stack=filter_stack)
    elif API_DUMP_FRAMEWORK_TYPE == "mindspore":
        ms_set_dump_switch("ON", mode, scope=scope, api_list=scope, filter_switch=filter_switch,
                           dump_type=dump_type, filter_stack=filter_stack)
    else:
        raise RuntimeError("You must call 'troubleshooter.api_dump.init' before calling"
                           "'troubleshooter.api_dump.start' function.")


def api_dump_stop():
    if API_DUMP_FRAMEWORK_TYPE == "torch":
        pt_set_dump_switch("OFF")
    elif API_DUMP_FRAMEWORK_TYPE == "mindspore":
        ms_set_dump_switch("OFF")
    else:
        raise RuntimeError("You must call 'troubleshooter.api_dump.init' before calling"
                           "'troubleshooter.api_dump.stop' function.")
