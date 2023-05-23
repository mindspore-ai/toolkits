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

import torch
from troubleshooter.migrator.save.base_saver import SaveTensorBase


class _SaveTensorTroch(SaveTensorBase):
    def __init__(self):
        super().__init__()
        self._cnt = 0

    def _clear_cnt(self):
        self._cnt = 0

    def _numpy(self, data):
        if torch.is_tensor(data):
            return data.cpu().detach().numpy()
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, " \
                            f"but got {type(data)}")

    def _shape(self, data):
        if torch.is_tensor(data):
            return tuple(data.shape)
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, " \
                            f"but got {type(data)}")

    def __call__(self, file, data, auto_id=True, suffix=None):
        path, name = self._handle_path(file)
        self._save_tensors(path, name, data, auto_id, suffix)
        if auto_id:
            self._cnt += 1
        return None


save = _SaveTensorTroch()
