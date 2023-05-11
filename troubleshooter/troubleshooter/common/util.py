# Copyright 2022 Tiger Miao and collaborators.
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
"""util functions"""
import re
import os
import stat
from collections import OrderedDict

from mindspore import Tensor, Parameter, nn
import mindspore as ms
import torch
import numpy as np


def print_line(char, times):
    print(char * times)


def get_ms_log_path():
    log_path = os.environ.get('GLOG_log_dir')
    return log_path


def create_save_path(data_save_path):
    """Get output path of data."""
    data_save_path = validate_and_normalize_path(data_save_path)
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path, exist_ok=True)
        os.chmod(data_save_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return data_save_path


def validate_and_normalize_path(
        path,
        check_absolute_path=False,
        allow_parent_dir=False,
):
    """
    Validates path and returns its normalized form.

    If path has a valid scheme, treat path as url, otherwise consider path a
    unix local path.

    Note:
        File scheme (rfc8089) is currently not supported.

    Args:
        path (str): Path to be normalized.
        check_absolute_path (bool): Whether check path scheme is supported.
        allow_parent_dir (bool): Whether allow parent dir in path.

    Returns:
        str, normalized path.
    """
    if not path:
        raise ValueError("The path is invalid!")

    path_str = str(path)
    if not allow_parent_dir:
        path_components = path_str.split("/")
        if ".." in path_components:
            raise ValueError("The path is invalid!")

    # path does not have valid schema, treat it as unix local path.
    if check_absolute_path:
        if not path_str.startswith("/"):
            raise ValueError("The path is invalid!")
    try:
        # most unix systems allow
        normalized_path = os.path.realpath(path)
    except ValueError:
        raise ValueError("The path is invalid!")

    return normalized_path


def compare_key_words(err_msg, key_word):
    """Find out whether the key log information is in the exception or not."""
    return key_word and err_msg and re.search(key_word, err_msg)


def compare_stack_words(err_msg, key_word):
    """Find out whether the key log information is in the exception or not."""
    if key_word and err_msg and not re.search(key_word, err_msg):
        return False
    return True


def find_file(dir, suffix=".npy"):
    file_list = []
    normal_dir = validate_and_normalize_path(dir)
    walk_generator = os.walk(normal_dir)
    for root_path, dirs, files in walk_generator:
        if len(files) < 1:
            continue
        for file in files:
            file_name, suffix_name = os.path.splitext(file)
            if suffix_name == suffix:
                file_list.append(file)
    return file_list


def make_directory(path: str):
    """Make directory."""
    if path is None or not isinstance(path, str) or path.strip() == "":
        raise TypeError("Input path '{}' is invalid type".format(path))

    path = os.path.realpath(path)

    if os.path.exists(path):
        real_path = path
    else:
        try:
            permissions = os.R_OK | os.W_OK | os.X_OK
            os.umask(permissions << 3 | permissions)
            mode = permissions << 6
            os.makedirs(path, mode=mode, exist_ok=True)
            real_path = path
        except PermissionError:
            raise TypeError("No write permission on the directory `{path}`.")
    return real_path


class SaveNet(nn.Cell):
    """
    The SaveNet class is used to build a unified data storage interface that supports PyTorch and MindSpore
    PYNATIVE_MODE as well as GRAPH_MODE, but currently does not support MindSpore GRAPH_MODE.

    Inputs:
        file (str): The name of the file to be stored.
        data (Union(Tensor)): Supports data types of Tensor for both MindSpore and PyTorch.
        auto_id (bool): Whether to enable automatic numbering. If set to True, an incremental number will be
          added before the saved file name. If set to False, no numbering will be added to the file name.
        suffix (str): The suffix of the saved file name.

    Outputs:
        The output storage name is '{id}_name_{suffix}.npy'.
    """

    def __init__(self):
        super(SaveNet, self).__init__()
        self.cnt = Parameter(Tensor(0, ms.int32),
                             name="cnt", requires_grad=False)
        self.sep = os.sep

    def numpy(self, data):
        if isinstance(data, ms.Tensor):
            return data.asnumpy()
        elif torch.is_tensor(data):
            return data.cpu().detach().numpy()
        else:
            raise TypeError(f"For ts.save, the type of argument 'data' must be mindspore.Tensor or torch.tensor, " \
                            f"but got {type(data)}")

    def handle_path(self, file):
        if file[-1] == self.sep:
            raise ValueError(f"For ts.save, the type of argument 'file' must be a valid filename, but got {file}")
        name = ''
        for c in file:
            if c == self.sep:
                name = ''
            else:
                name += c
        path = ''
        for i in range(len(file) - len(name)):
            path += file[i]
        return path, name

    def construct(self, file, data, auto_id, suffix):
        path, name = self.handle_path(file)
        if auto_id:
            np.save(f"{path}{int(self.cnt)}_{name}_{suffix}" if suffix else
                    f"{path}{int(self.cnt)}_{name}", self.numpy(data))
        else:
            np.save(f"{file}_{suffix}" if suffix else file,
                    self.numpy(data))
        if auto_id:
            self.cnt += 1
        return

save = SaveNet()

class _SaveNet(nn.Cell):
    """
    The SaveNet class is used to build a unified data storage interface that supports PyTorch and MindSpore
    PYNATIVE_MODE as well as GRAPH_MODE, but currently does not support MindSpore GRAPH_MODE.

    Inputs:
        file (str): The name of the file to be stored.
        data (Union(Tensor, list[Tensor], Tuple[Tensor], dict[str, Tensor])): Supports data types of Tensor,
          list[Tensor], tuple(Tensor), and dict[str, Tensor] for both MindSpore and PyTorch. When the input is
          a list or tuple of Tensor, the file name will be numbered according to the index of the Tensor.
          When the input is a dictionary of Tensor, the corresponding key will be added to the file name.
        auto_id (bool): Whether to enable automatic numbering. If set to True, an incremental number will be
          added before the saved file name. If set to False, no numbering will be added to the file name.
        suffix (str): The suffix of the saved file name.

    Outputs:
        The output storage name is '{id}_name_{idx/key}_{suffix}.npy'.
    """

    def __init__(self):
        super(SaveNet, self).__init__()
        self.cnt = Parameter(Tensor(0, ms.int32),
                             name="cnt", requires_grad=False)
        self.sep = os.sep

    def numpy(self, data):
        if isinstance(data, ms.Tensor):
            return data.asnumpy()
        elif torch.is_tensor(data):
            return data.cpu().detach().numpy()
        else:
            raise TypeError(f"For ts.save, the type of argument 'data' must be mindspore.Tensor or torch.tensor, " \
                            f"but got {type(data)}")

    def handle_path(self, file):
        if file[-1] == self.sep:
            raise ValueError(f"For ts.save, the type of argument 'file' must be a valid filename, but got {file}")
        name = ''
        for c in file:
            if c == self.sep:
                name = ''
            else:
                name += c
        path = ''
        for i in range(len(file) - len(name)):
            path += file[i]
        return path, name

    def construct(self, file, data, auto_id, suffix):
        path, name = self.handle_path(file)
        if isinstance(data, (list, tuple)):
            for idx, val in enumerate(data):
                if auto_id:
                    np.save(f"{path}{int(self.cnt)}_{name}_{idx}_{suffix}" if suffix else
                            f"{path}{int(self.cnt)}_{name}_{idx}", self.numpy(val))
                else:
                    np.save(f"{file}_{idx}_{suffix}" if suffix else
                            f"{file}_{idx}", self.numpy(val))
        elif isinstance(data, (dict, OrderedDict)):
            for key, val in data.items():
                if auto_id:
                    np.save(f"{path}{int(self.cnt)}_{name}_{key}_{suffix}" if suffix else
                            f"{path}{int(self.cnt)}_{name}_{key}", self.numpy(val))
                else:
                    np.save(f"{file}_{key}_{suffix}" if suffix else
                            f"{file}_{key}", self.numpy(val))
        else:
            if auto_id:
                np.save(f"{path}{int(self.cnt)}_{name}_{suffix}" if suffix else
                        f"{path}{int(self.cnt)}_{name}", self.numpy(data))
            else:
                np.save(f"{file}_{suffix}" if suffix else file,
                        self.numpy(data))
        if auto_id:
            self.cnt += 1


_save = _SaveNet()


def save_numpy_data(file_path, data):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    np.save(file_path, data)
