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

from .. import universal_interface

cell_count = defaultdict(int)
g_stop_hook = False


class HOOKCell(nn.Cell):

    def __init__(self, hook) -> None:
        super(HOOKCell, self).__init__()
        self.changed_status = False

        global g_stop_hook
        if not g_stop_hook:
            g_stop_hook = True
            prefix = ""
            self.changed_status = True
            if hasattr(self, "prefix_op_name_"):
                prefix = self.prefix_op_name_

            cell_count[prefix] += 1
            prefix = prefix + str(cell_count[prefix] - 1) + '_'
            self.register_forward_hook(hook(prefix + "forward"))
            if universal_interface.g_retain_backward:
                self.register_backward_hook(hook(prefix + "backward"))

    # 重载call，加全局标志。
    def __call__(self, *args, **kwargs):
        try:
            out = super(HOOKCell, self).__call__(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            if self.changed_status:
                self.changed_status = False
                global g_stop_hook
                g_stop_hook = False
        return out
