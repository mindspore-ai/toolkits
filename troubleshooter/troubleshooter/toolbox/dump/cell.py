# Copyright 2023 Huawei Technologies Co., Ltd
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
from collections import defaultdict
from mindspore import nn

cell_count = defaultdict(int)
g_stop_hook = False


class HOOKCell(nn.Cell):

    def __init__(self, hook) -> None:
        super(HOOKCell, self).__init__()
        self.input_args = tuple()
        self.input_kwargs = dict()
        prefix = ""
        if hasattr(self, "prefix_op_name_"):
            prefix = self.prefix_op_name_

        cell_count[prefix] += 1
        prefix = prefix + str(cell_count[prefix] - 1) + '_'

        self.register_forward_hook(hook(prefix + "forward", forward=True))
        self.register_backward_hook(hook(prefix + "backward", forward=False))

    # 重载call，加全局标志。
    def __call__(self, *args, **kwargs):
        changed = False
        global g_stop_hook
        if g_stop_hook:
            self._enable_forward_hook = False
            self._enable_backward_hook = False
        else:
            g_stop_hook = True
            changed = True
        out = super(HOOKCell, self).__call__(*args, **kwargs)
        if changed:
            g_stop_hook = False
        return out
