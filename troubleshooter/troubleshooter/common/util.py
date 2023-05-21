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
"""util functions"""
import os
import re
import stat
from collections import OrderedDict

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


def save_numpy_data(file_path, data):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    np.save(file_path, data)


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


def iterate_items(data):
    if isinstance(data, (dict, OrderedDict)):
        return data.items()
    elif isinstance(data, (list, tuple)):
        return enumerate(data)
    else:
        raise TypeError("Unsupported data type")
