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
import stat
import numpy as np
from troubleshooter import FRAMEWORK_TYPE

TORCH_SAVE_COUNT = 0
SAVE_NAME_MARK = "_TS_SAVE_NAME:"


def _split_path_and_name(file, sep=os.sep):
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


def _remove_npy_extension(file_name):
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


def _check_path_type(file):
    if isinstance(file, str):
        return True
    else:
        raise TypeError(
            f"For 'ts.save', the type of argument 'file' must be str, but got {type(file)}")


def _check_save_mode(mode, func_name):
    save_mode = ['npy', 'print']
    if mode not in save_mode:
        raise ValueError(f'Fot {func_name}, the output_mode must in {save_mode}, but got output_mode is {mode}.')


def _handle_path(file):
    if file:
        _check_path_type(file)
        path, name = _split_path_and_name(file)
        name = _remove_npy_extension(name)
    else:
        path, name = "", ""
    return path, name


def _iterate_items(data):
    if isinstance(data, (dict, OrderedDict)):
        return data.items()
    elif isinstance(data, (list, tuple)):
        return enumerate(data)
    else:
        raise TypeError("Unsupported data type")


def _add_id_prefix_to_filename(filename):
    global TORCH_SAVE_COUNT
    new_filename = f"{TORCH_SAVE_COUNT}_{filename}"
    TORCH_SAVE_COUNT += 1
    return new_filename


class _SaveBase:
    def __init__(self, file):
        super(_SaveBase, self).__init__()
        path, name = _handle_path(file)
        self.path = path
        self.name = name
        self.save_func = {'npy': _npy_save, 'print': _print_save}

    def get_save_func(self, mode):
        return self.save_func[mode]


class _SaveGradBase:
    def __init__(self, file, suffix, output_mode):
        super(_SaveGradBase, self).__init__()
        path, name = _handle_path(file)
        if suffix:
            name = f"{name}_{suffix}"
        if output_mode == 'print':
            self.file = name
        else:
            self.file = f"{path}{name}"


def torch_TensorDump(file, data):
    directory, filename = os.path.split(file)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, mode=stat.S_IRWXU, exist_ok=True)
    filename = _add_id_prefix_to_filename(filename)
    file = os.path.join(directory, filename)
    if not file.endswith(".npy"):
        file = file + ".npy"
    if os.path.exists(file):
        os.chmod(file, stat.S_IWUSR)
    np.save(file, data.cpu().detach().numpy())
    os.chmod(file, stat.S_IRUSR)


def _wrapper_torch_save_grad(file, output_mode):
    def _save_grad_func(grad):
        if grad is None:
            return
        if output_mode == 'print':
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
            if data.dtype == ms.bfloat16:
                data = data.float()
            ms.ops.TensorDump()(file, data)
        elif isinstance(data, torch.Tensor):
            torch_TensorDump(file, data)
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, "
                            f"but got {type(data)}")

    def _wrapper_save_grad_func(file, output_mode):
        def _save_grad_func(grad):
            if output_mode == 'print':
                format_name = f"{SAVE_NAME_MARK}{file}"
                print(format_name, grad)
            else:
                data = grad
                if data.dtype == ms.bfloat16:
                    data = data.float()
                ms.ops.TensorDump()(file, data)
            return grad
        return _save_grad_func

    @ms.jit_class
    class _SaveGradCell(_SaveGradBase):
        def __init__(self, file, suffix, output_mode):
            super(_SaveGradCell, self).__init__(file, suffix, output_mode)
            self.ms_save_grad = ms.ops.InsertGradientOf(
                _wrapper_save_grad_func(self.file, output_mode))
            self.pt_save_func = _wrapper_torch_save_grad(self.file, output_mode)

        def __call__(self, x):
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
        if isinstance(data, torch.Tensor):
            torch_TensorDump(file, data)
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, "
                            f"but got {type(data)}")

    class _SaveGradCell(_SaveGradBase):
        def __init__(self, file, suffix, output_mode):
            super(_SaveGradCell, self).__init__(file, suffix, output_mode)
            self.pt_save_func = _wrapper_torch_save_grad(self.file, output_mode)

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
            if data.dtype == ms.bfloat16:
                data = data.float()
            ms.ops.TensorDump()(file, data)
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, "
                            f"but got {type(data)}")

    def _wrapper_save_grad_func(file, output_mode):
        def _save_grad_func(grad):
            if output_mode == 'print':
                format_name = f"{SAVE_NAME_MARK}{file}"
                print(format_name, grad)
            else:
                data = grad
                if data.dtype == ms.bfloat16:
                    data = data.float()
                ms.ops.TensorDump()(file, data)
            return grad
        return _save_grad_func

    @ms.jit_class
    class _SaveGradCell(_SaveGradBase):
        def __init__(self, file, suffix, output_mode):
            super(_SaveGradCell, self).__init__(file, suffix, output_mode)
            self.ms_save_grad = ms.ops.InsertGradientOf(
                _wrapper_save_grad_func(self.file, output_mode))

        def __call__(self, x):
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

    @ms.jit_class
    class _SaveCell(_SaveBase):
        def __call__(self, data, suffix, output_mode):
            self.get_save_func(output_mode)(self.name, data, suffix, self.path)

else:
    class _SaveCell(_SaveBase):
        def __call__(self, data, suffix, output_mode):
            self.get_save_func(output_mode)(self.name, data, suffix, self.path)


def _npy_save(item_name, data, suffix, path):
    if isinstance(data, (list, tuple, dict, OrderedDict)):
        for key, val in _iterate_items(data):
            _npy_save(f"{item_name}.{key}", val, suffix, path)
    else:
        if data is None:
            return
        if suffix:
            item_name = f"{item_name}_{suffix}"
        _npy_save_ops(f"{path}{item_name}", data)


def _print_save(item_name, data, suffix, path=None):
    if isinstance(data, (list, tuple, dict, OrderedDict)):
        for key, val in _iterate_items(data):
            _print_save(f"{item_name}.{key}", val, suffix, path)
    else:
        if data is None:
            return
        if suffix:
            item_name = f"{item_name}_{suffix}"
        format_name = f"{SAVE_NAME_MARK}{item_name}"
        print(format_name, data)


def save(file, data, suffix=None, output_mode="npy"):
    """
    save tensor.
    """
    _check_save_mode(output_mode, "save")
    _SaveCell(file)(data, suffix, output_mode)


def save_grad(file, data, suffix="backward", output_mode="npy"):
    """
    save grad.
    """
    _check_save_mode(output_mode, "save_grad")
    return _SaveGradCell(file, suffix, output_mode)(data)
