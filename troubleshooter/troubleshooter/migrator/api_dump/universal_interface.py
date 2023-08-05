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

if "torch" in FRAMEWORK_TYPE:
    import torch

if "mindspore" in FRAMEWORK_TYPE:
    import mindspore

API_DUMP_FRAMEWORK_TYPE = None
g_retain_backward  = None

def api_dump_init(net, output_path=os.path.join(os.getcwd(), "ts_api_dump"), *, retain_backward=False):
    global API_DUMP_FRAMEWORK_TYPE
    global g_retain_backward
    if "torch" in FRAMEWORK_TYPE and isinstance(net, torch.nn.Module):
        from troubleshooter.migrator.api_dump.pt_dump import (acc_cmp_dump, register_hook,
                                                              set_dump_path)
        API_DUMP_FRAMEWORK_TYPE = "torch"
        set_dump_path(output_path)
    elif "mindspore" in FRAMEWORK_TYPE and isinstance(net, mindspore.nn.Cell):
        API_DUMP_FRAMEWORK_TYPE = "mindspore"
        from troubleshooter.migrator.api_dump.ms_dump import (acc_cmp_dump, register_hook,
                                                              set_dump_path)
        set_dump_path(output_path)
    else:
        raise TypeError(f"For 'troubleshooter.api_dump.init' function, the type of argument 'net' must be mindspore.nn.Cell or torch.nn.Module, but got {type(net)}.")
    g_retain_backward = retain_backward
    register_hook(net, acc_cmp_dump)

def api_dump_start():
    if API_DUMP_FRAMEWORK_TYPE == "torch":
        from troubleshooter.migrator.api_dump.pt_dump import set_dump_switch
    elif API_DUMP_FRAMEWORK_TYPE == "mindspore":
        from troubleshooter.migrator.api_dump.ms_dump import set_dump_switch
    else:
        raise RuntimeError("You must call 'troubleshooter.api_dump.init' before calling"
                           "'troubleshooter.api_dump.start' function.")
    set_dump_switch("ON")

def api_dump_stop():
    if API_DUMP_FRAMEWORK_TYPE == "torch":
        from troubleshooter.migrator.api_dump.ms_dump import set_dump_switch
    elif API_DUMP_FRAMEWORK_TYPE == "mindspore":
        from troubleshooter.migrator.api_dump.pt_dump import set_dump_switch
    else:
        raise RuntimeError("You must call 'troubleshooter.api_dump.init' before calling"
                           "'troubleshooter.api_dump.stop' function.")
    set_dump_switch("OFF")
