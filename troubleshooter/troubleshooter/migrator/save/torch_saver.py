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

import os
from collections import OrderedDict

import numpy as np
import torch
from troubleshooter.common.util import iterate_items, remove_npy_extension, split_path_and_name

_g_save_cnt = 0


def save(file, data):
    def numpy(data):
        if torch.is_tensor(data):
            return data.cpu().detach().numpy()
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, " \
                            f"but got {type(data)}")

    np_data = numpy(data)
    if file:
        path, name = split_path_and_name(file)
        name = remove_npy_extension(name)
    else:
        path, name = '', 'tensor_' + str(tuple(data.shape))
    if not name:
        raise ValueError(f"For 'ts.save', the type of argument 'file' must be a valid filename, but got {file}")
    global _g_save_cnt
    np.save(os.path.join(path, f"{_g_save_cnt}_{name}"), np_data)
    _g_save_cnt += 1


def _save(self, file, data, auto_id=True, suffix=None):
    def numpy(data):
        if torch.is_tensor(data):
            return data.cpu().detach().numpy()
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, " \
                            f"but got {type(data)}")

    if file:
        path, name = split_path_and_name(file)
        name = remove_npy_extension(name)
    else:
        path, name = '', 'tensor_' + str(tuple(data.shape))
    global _g_save_cnt
    if isinstance(data, (list, tuple, dict, OrderedDict)):
        for key, val in iterate_items(data):
            if auto_id:
                np.save(f"{path}{int(_g_save_cnt)}_{name}_{key}_{suffix}" if suffix else
                        f"{path}{int(_g_save_cnt)}_{name}_{key}", numpy(val))
            else:
                np.save(f"{path}{name}_{key}_{suffix}" if suffix else
                        f"{path}{name}_{key}", numpy(val))
    else:
        if auto_id:
            np.save(f"{path}{int(_g_save_cnt)}_{name}_{suffix}" if suffix else
                    f"{path}{int(_g_save_cnt)}_{name}", numpy(data))
        else:
            np.save(f"{path}{name}_{suffix}" if suffix else file, numpy(data))
    if auto_id:
        _g_save_cnt += 1
    return None
