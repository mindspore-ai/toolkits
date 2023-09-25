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
from builtins import print as builtin_print

import torch.nn as nn
import torch.optim as optim
import yaml

from ..common import global_manage

cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")

with open(yaml_path, "r") as f:
    WrapModuleOps = yaml.safe_load(f).get("Module")

NNModule = {}

for f in dir(nn):
    NNModule[f] = getattr(nn, f)


def get_nn_module():
    global WrapModuleOps
    _all_nn_module = dir(nn)
    return set(WrapModuleOps) & set(_all_nn_module)


def stop_dump_hook(func):
    def wrapper(*args, **kwargs):
        if not global_manage.get_value("g_stop_hook"):
            global_manage.set_value("g_stop_hook", True)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                raise e
            finally:
                global_manage.set_value("g_stop_hook", False)
        else:
            return func(*args, **kwargs)
    return wrapper


def call_decorator(cls, name):
    cls.hook_name = "wrap_" + name

    cls.__call__ = stop_dump_hook(cls.__call__)
    return cls


def remove_dropout_randomness(cls):
    def new_forward(self, x):
        return x

    cls.forward = new_forward
    return cls


def wrap_nn_module_and_bind():
    _nn_module = get_nn_module()
    for name in _nn_module:
        if name.startswith("Dropout"):
            remove_dropout_randomness(NNModule[name])
        call_decorator(NNModule[name], name)


def wrap_optimizer():
    for cls_name in dir(optim):
        cls = getattr(optim, cls_name)
        if cls_name != 'Optimizer' and isinstance(cls, type) and issubclass(cls, optim.Optimizer):
            cls.step = stop_dump_hook(cls.step)
            cls.zero_grad = stop_dump_hook(cls.zero_grad)


def stop_hook_print(*args, **kwargs):
    if not global_manage.get_value("g_stop_hook"):
        global_manage.set_value("g_stop_hook", True)
        try:
            return builtin_print(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            global_manage.set_value("g_stop_hook", False)
    else:
        return builtin_print(*args, **kwargs)


def wrap_lr_scheduler(cls):
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.optimizer.step = stop_dump_hook(self.optimizer.step)

    cls.__init__ = new_init
    return cls
