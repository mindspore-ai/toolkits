# Copyright 2023 David Fan and collaborators.
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
from troubleshooter import FRAMEWORK_TYPE


def split_path_and_name(file, sep=os.sep):
    if file[-1] == sep:
        raise ValueError(f"For 'ts.save', the type of argument 'file' must be a valid filename, but got {file}")
    path, name = "", ""
    for c in file:
        if c == sep:
            path = path + name + sep
            name = ""
        else:
            name += c
    return path, name


def remove_npy_extension(file_name):
    has_extension = False
    extension = ""
    file_name_without_extension = ""

    for char in file_name:
        if char == ".":
            file_name_without_extension += extension
            has_extension = True
            extension = "."
        elif has_extension:
            extension += char
        else:
            file_name_without_extension += char

    if extension == ".npy":
        return file_name_without_extension
    else:
        return file_name


def check_path_type(file):
    if isinstance(file, str):
        return True
    else:
        raise TypeError(f"For 'ts.save', the type of argument 'file' must be str, but got {type(file)}")
    
def handle_path(file):
    if file:
        check_path_type(file)
        path, name = split_path_and_name(file)
        name = remove_npy_extension(name)
    else:
        path, name = "", ""
    return path, name


def iterate_items(data):
    if isinstance(data, (dict, OrderedDict)):
        return data.items()
    elif isinstance(data, (list, tuple)):
        return enumerate(data)
    else:
        raise TypeError("Unsupported data type")


if {"torch", "mindspore"}.issubset(FRAMEWORK_TYPE):
    import torch
    import mindspore as ms


    def numpy(data):
        if isinstance(data, ms.Tensor):
            return data.asnumpy()
        elif torch.is_tensor(data):
            return data.cpu().detach().numpy()
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, " \
                            f"but got {type(data)}")


    def shape(data):
        if isinstance(data, ms.Tensor):
            return data.shape
        elif torch.is_tensor(data):
            return tuple(data.shape)
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, " \
                            f"but got {type(data)}")
elif "torch" in FRAMEWORK_TYPE:
    import torch


    def numpy(data):
        if torch.is_tensor(data):
            return data.cpu().detach().numpy()
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, " \
                            f"but got {type(data)}")


    def shape(data):
        if torch.is_tensor(data):
            return tuple(data.shape)
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, " \
                            f"but got {type(data)}")
elif "mindspore" in FRAMEWORK_TYPE:
    import mindspore as ms


    def numpy(data):
        if isinstance(data, ms.Tensor):
            return data.asnumpy()
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, " \
                            f"but got {type(data)}")


    def shape(data):
        if isinstance(data, ms.Tensor):
            return data.shape
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, " \
                            f"but got {type(data)}")
else:
    def numpy(data):
        return data


    def shape(data):
        return data.shape

if "mindspore" in FRAMEWORK_TYPE:
    import mindspore as ms


    @ms.jit_class
    class TsSaveCount:
        def __init__(self):
            self.cnt = ms.Parameter(ms.Tensor(-1, ms.int32), name="ts_save_cnt", requires_grad=False)

        def add_one(self):
            self.cnt += 1
            return self.cnt

        def reset(self):
            self.cnt = ms.Parameter(ms.Tensor(-1, ms.int32), name="ts_save_cnt", requires_grad=False)

        def get(self):
            return self.cnt
else:
    class TsSaveCount:
        def __init__(self):
            self.cnt = -1

        def add_one(self):
            self.cnt += 1
            return self.cnt

        def reset(self):
            self.cnt = -1

        def get(self):
            return self.cnt

_ts_save_cnt = TsSaveCount()


def save(file, data, auto_id=True, suffix=None):
    path, name = handle_path(file)
    if auto_id:
        _ts_save_cnt.add_one()
    cnt = _ts_save_cnt.get()
    def _save(item_name, data):    
        if isinstance(data, (list, tuple, dict, OrderedDict)):
            for key, val in iterate_items(data):
                _save(f"{item_name}_{key}", val)
        else:
            if data is None:
                return
            if name == "":
                item_name = f"{item_name}_{shape(data)}"
            if auto_id:
                np.save(f"{path}{int(cnt)}_{item_name}_{suffix}" if suffix else
                        f"{path}{int(cnt)}_{item_name}", numpy(data))
            else:
                np.save(f"{path}{item_name}_{suffix}" if suffix else
                        f"{path}{item_name}", numpy(data))
    item_name = name if name else "tensor"
    _save(item_name, data)
    return None
