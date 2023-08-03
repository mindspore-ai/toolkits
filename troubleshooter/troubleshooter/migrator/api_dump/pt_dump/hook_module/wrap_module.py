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
import os
import torch.nn as nn
import yaml
from ..common import global_manage

cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")

with open(yaml_path, "r") as f:
    WrapModuleOps = yaml.safe_load(f).get("Module")

nn_module = {}

for f in dir(nn):
    nn_module[f] = getattr(nn, f)


def get_nn_module():
    global WrapModuleOps
    _all_nn_module = dir(nn)
    return set(WrapModuleOps) & set(_all_nn_module)


def call_decorator(cls, name):
    original_call = cls.__call__
    cls.hook_name = "wrap_" + name

    def new_call(self, *args, **kwargs):
        changed = False
        if not global_manage.get_value("g_stop_hook"):
            global_manage.set_value("g_stop_hook", True)
            changed = True

        result = original_call(self, *args, **kwargs)
        if changed:
            global_manage.set_value("g_stop_hook", False)
        return result

    cls.__call__ = new_call
    return cls


def wrap_nn_module_and_bind():
    _nn_module = get_nn_module()
    for name in _nn_module:
        call_decorator(nn_module[name], name)
