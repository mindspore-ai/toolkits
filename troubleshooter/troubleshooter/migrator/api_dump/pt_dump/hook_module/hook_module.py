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
from collections import defaultdict

import torch.nn as nn

from ..common import global_manage

module_count = defaultdict(int)


class HOOKModule(nn.Module):

    def __init__(self, hook) -> None:
        super(HOOKModule, self).__init__()
        self.changed_status = False
        if not global_manage.get_value("g_stop_hook"):
            global_manage.set_value("g_stop_hook", True)
            prefix = ""
            if hasattr(self, "prefix_op_name_"):
                prefix = self.prefix_op_name_
            self.changed_status = True

            module_count[prefix] += 1
            prefix = prefix + str(module_count[prefix] - 1) + '_'

            self.register_forward_hook(hook(prefix + "forward"))

    def __call__(self, *input, **kwargs):
        try:
            result = super(HOOKModule, self).__call__(*input, **kwargs)
        except Exception as e:
            raise e
        finally:
            if self.changed_status:
                self.changed_status = False
                global_manage.set_value("g_stop_hook", False)
        return result
