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
import re
from collections import OrderedDict
from pathlib import Path
import stat

import numpy as np

from troubleshooter import FRAMEWORK_TYPE
from troubleshooter import log as logger
from troubleshooter.common.util import isfile_check, validate_and_normalize_path

TORCH_SAVE_COUNT = 0
SAVE_NAME_MARK = "_TS_SAVE_NAME:"


def split_path_and_name(file, sep=os.sep):
    if file[-1] == sep:
        raise ValueError(
            f"For 'ts.save', the type of argument 'file' must be a valid filename, but got {file}")
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
        raise TypeError(
            f"For 'ts.save', the type of argument 'file' must be str, but got {type(file)}")


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


def add_id_prefix_to_filename(original_path):
    path, filename = os.path.split(original_path)
    global TORCH_SAVE_COUNT
    new_filename = f"{TORCH_SAVE_COUNT}_{filename}"
    TORCH_SAVE_COUNT += 1
    new_path = os.path.join(path, new_filename)
    return new_path


def torch_TensorDump(file, data):
    file = add_id_prefix_to_filename(file)
    if not file.endswith(".npy"):
        file = file + ".npy"
    np.save(file, data.cpu().detach().numpy())
    os.chmod(file, stat.S_IRUSR)


def _wrapper_torch_save_grad(file, use_print):
    def _save_grad_func(grad):
        if use_print:
            format_name = f"{SAVE_NAME_MARK}{file}"
            print(format_name, grad)
        else:
            torch_TensorDump(file, grad)
        return grad
    return _save_grad_func


if {"torch", "mindspore"}.issubset(FRAMEWORK_TYPE):
    import torch
    import mindspore as ms

    def _npy_save_ops(file, data):
        if isinstance(data, ms.Tensor):
            ms.ops.TensorDump()(file, data)
        elif torch.is_tensor(data):
            torch_TensorDump(file, data)
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, "
                            f"but got {type(data)}")

    def _wrapper_save_grad_func(file, use_print):
        def _save_grad_func(grad):
            if use_print:
                format_name = f"{SAVE_NAME_MARK}{file}"
                print(format_name, grad)
            else:
                ms.ops.TensorDump()(file, grad)
            return grad
        return _save_grad_func

    class SaveGradCell(ms.nn.Cell):
        def __init__(self, file, suffix="backward", use_print=False):
            super(SaveGradCell, self).__init__()
            path, name = handle_path(file)
            if suffix:
                name = f"{name}_{suffix}"
            if use_print:
                file = name
            else:
                file = f"{path}{name}"
            self.ms_save_grad = ms.ops.InsertGradientOf(
                _wrapper_save_grad_func(file, use_print))
            self.pt_save_func = _wrapper_torch_save_grad(file, use_print)

        def construct(self, x):
            if isinstance(x, ms.Tensor):
                return self.ms_save_grad(x)
            elif isinstance(x, torch.Tensor):
                x.register_hook(self.pt_save_func)
                return x
            else:
                raise TypeError(f"For 'ts.save_grad', the type of argument 'data' must be mindspore.Tensor or torch.tensor, "
                                f"but got {type(x)}")


elif "torch" in FRAMEWORK_TYPE:
    import torch

    def _npy_save_ops(file, data):
        if torch.is_tensor(data):
            torch_TensorDump(file, data)
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, "
                            f"but got {type(data)}")

    class SaveGradCell:
        def __init__(self, file, suffix="backward", use_print=False):
            path, name = handle_path(file)
            if suffix:
                name = f"{name}_{suffix}"
            if use_print:
                file = name
            else:
                file = f"{path}{name}"
            self.pt_save_func = _wrapper_torch_save_grad(file, use_print)

        def __call__(self, x):
            if isinstance(x, torch.Tensor):
                x.register_hook(self.pt_save_func)
                return x
            else:
                raise TypeError(f"For 'ts.save_grad', the type of argument 'data' must be mindspore.Tensor or torch.tensor, "
                                f"but got {type(x)}")
elif "mindspore" in FRAMEWORK_TYPE:
    import mindspore as ms

    def _npy_save_ops(file, data):
        if isinstance(data, ms.Tensor):
            ms.ops.TensorDump()(file, data)
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, "
                            f"but got {type(data)}")

    def _wrapper_save_grad_func(file, use_print):
        def _save_grad_func(grad):
            if use_print:
                format_name = f"{SAVE_NAME_MARK}{file}"
                print(format_name, grad)
            else:
                ms.ops.TensorDump()(file, grad)
            return grad
        return _save_grad_func

    class SaveGradCell(ms.nn.Cell):
        def __init__(self, file, suffix="backward", use_print=False):
            super(SaveGradCell, self).__init__()
            path, name = handle_path(file)
            if suffix:
                name = f"{name}_{suffix}"
            if use_print:
                file = name
            else:
                file = f"{path}{name}"
            self.ms_save_grad = ms.ops.InsertGradientOf(
                _wrapper_save_grad_func(file, use_print))

        def construct(self, x):
            if isinstance(x, ms.Tensor):
                return self.ms_save_grad(x)
            else:
                raise TypeError(f"For 'ts.save_grad', the type of argument 'data' must be mindspore.Tensor or torch.tensor, "
                                f"but got {type(x)}")
else:
    def numpy(data):
        return data

if "mindspore" in FRAMEWORK_TYPE:
    import mindspore as ms

    class _ConvertData:
        """
        Offline conversion of files saved by ts.save to npy files.
        Using a finite state machine, state switching is as follows.
        +------------+----------------------------------------------+---------------------------------------------+
        | now \ next | pepare                                       | parse_data                                  |
        +============+==============================================+=============================================+
        | pepare     | condition: normal                            | condition: input starts with SAVE_NAME_MARK |
        |            | action: skip                                 | action: parse_data                          |
        | parse_data | condition: parsing completed or illegal data | no exist                                    |
        |            | action:reset                                 | no exist                                    |
        +------------+----------------------------------------------+---------------------------------------------+
        """
        tensor_to_np_type = {"Int8": np.int8, "UInt8": np.uint8, "Int16": np.int16, "UInt16": np.uint16,
                             "Int32": np.int32, "UInt32": np.uint32, "Int64": np.int64, "UInt64": np.uint64,
                             "Float16": np.float16, "Float32": np.float32, "Float64": np.float64, "Bool": np.bool_, "str": "U"}

        def __init__(self, output_path):
            self.states = ['pepare', 'parse_data']
            self.state = self.states[0]
            self.path = Path()
            self.output_path = Path(output_path)
            if not self.output_path.exists():
                self.output_path.mkdir(mode=0o700)
            self.name = ""
            self.auto_id = -1

            self.events = dict()
            self.actions = {state: dict() for state in self.states}
            self._add_event('pepare', self.pepare)
            self._add_action('pepare', 'pepare', self.skip)
            self._add_action('pepare', 'parse_data', self.parse_name)
            self._add_event('parse_data', self.parse_data)
            self._add_action('parse_data', 'pepare', self.reset)

        def _add_event(self, state, handler):
            self.events[state] = handler

        def _add_action(self, cur_state, next_state, handler):
            self.actions[cur_state][next_state] = handler

        def run(self, inputs):
            try:
                new_state, outputs = self.events[self.state](inputs)
                self.actions[self.state][new_state](outputs)
                self.state = new_state
            except Exception as e:
                logger.user_error("Because the data was incomplete when saving, "
                                  "the parsing of this data failed. "
                                  f"The current status is {self.state}, and the exception is {e}.")
                self.reset()
                new_state, outputs = 'pepare', None
            return new_state

        def pepare(self, inputs):
            if isinstance(inputs, str) and inputs.startswith(SAVE_NAME_MARK):
                return self.states[1], inputs
            return self.states[0], None

        def parse_name(self, data_info):
            self.name = data_info[len(SAVE_NAME_MARK):]
            self.auto_id += 1

        def parse_data(self, data):
            np_data = self._convert2ndarray(data)
            self._save_data(np_data)
            return self.states[0], None

        def _parse_shape(self, data):
            if data:
                return tuple(map(int, data.split()))
            else:
                return tuple()

        def _convert_str2ndarray(self, input_string):
            """
            Convert GE string Tensor to ndarray
            """
            pattern = r'Tensor\(shape=(.*?), dtype=(.*?), value=(.*?)\)'

            match = re.search(pattern, input_string, re.DOTALL)
            if match:
                shape_str = match.group(1).strip()
                dtype_str = match.group(2).strip()
                value_str = match.group(3).strip().replace(
                    '[', '').replace(']', '')
                shape = self._parse_shape(shape_str[1:-1])
                dtype = self.tensor_to_np_type[dtype_str]
                value = np.fromstring(
                    value_str, dtype=dtype, sep=' ').reshape(shape)
                return value
            raise ValueError(f"Data type error, miss Tensor")

        def _convert2ndarray(self, data):
            if isinstance(data, ms.Tensor):
                return data.asnumpy()
            elif isinstance(data, str):
                return self._convert_str2ndarray(data)
            else:
                raise ValueError(f"Data type error, got type is {type(data)}")

        def _save_data(self, data):
            self.name = str(self.auto_id) + "_" + self.name
            file_path = str(self.output_path/self.name)
            if not file_path.endswith(".npy"):
                file_path = file_path + ".npy"
            np.save(file_path, data)
            os.chmod(file_path, stat.S_IRUSR)

        def reset(self, *args, **kwargs):
            self.name = ""

        def skip(*args, **kwargs):
            return None

    def save_convert(file, output_path):
        isfile_check(file, 'file')
        validate_and_normalize_path(output_path)
        data = ms.parse_print(file)
        output_path = Path(output_path)
        converter = _ConvertData(output_path)
        logger.info("Print dump data total len is ", len(data))
        for item in data:
            converter.run(item)
        logger.user_attention(
            f"Convert data has been saved in {output_path.absolute()}")

    class SaveCell(ms.nn.Cell):
        def __init__(self, file):
            super(SaveCell, self).__init__()
            path, name = handle_path(file)
            self.path = path
            self.name = name

        def construct(self, data, suffix=None, use_print=False):
            if not use_print:
                _npy_save(self.name, data, suffix, self.path)
            else:
                _print_save(self.name, data, suffix)

else:
    def save_convert(file, output_path='npy_files'):
        raise ValueError("There is no 'mindspore' package in the current environment, "
                         "and 'convert_data' does not support calling.")

    class SaveCell:
        def __init__(self, file):
            path, name = handle_path(file)
            self.path = path
            self.name = name

        def __call__(self, data, suffix=None, use_print=False):
            if not use_print:
                _npy_save(self.name, data, suffix, self.path)
            else:
                _print_save(self.name, data, suffix)


def _npy_save(item_name, data, suffix, path):
    if isinstance(data, (list, tuple, dict, OrderedDict)):
        for key, val in iterate_items(data):
            _npy_save(f"{item_name}.{key}", val, suffix, path)
    else:
        if data is None:
            return
        if suffix:
            item_name = f"{item_name}_{suffix}"
        _npy_save_ops(f"{path}{item_name}", data)


def _print_save(item_name, data, suffix):
    if isinstance(data, (list, tuple, dict, OrderedDict)):
        for key, val in iterate_items(data):
            _print_save(f"{item_name}.{key}", val, suffix)
    else:
        if data is None:
            return
        if suffix:
            item_name = f"{item_name}_{suffix}"
        format_name = f"{SAVE_NAME_MARK}{item_name}"
        print(format_name, data)


def save(file, data, suffix=None, use_print=False):
    """
    save tensor to npy file.
    """
    SaveCell(file)(data, suffix, use_print)


def save_grad(file, data, suffix="backward", use_print=False):
    return SaveGradCell(file, suffix, use_print)(data)
