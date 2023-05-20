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

import mindspore
class StopMonitor(mindspore.train.callback.Callback):
    def __init__(self, stop_epoch=1, stop_step=1):
        """定义初始化过程"""
        super(StopMonitor, self).__init__()
        self.stop_epoch = stop_epoch  # 定义step
        self.stop_step = stop_step  # 定义step

    def on_train_step_end(self, run_context):
        """每个step结束后执行的操作"""
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num  # 获取epoch值
        step_num = cb_params.cur_step_num  # 获取step值
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        if step_num == 1875:
            print(step_num)
        if epoch_num == self.stop_epoch and cur_step_in_epoch == self.stop_step:
            run_context.request_stop()  # 停止训练