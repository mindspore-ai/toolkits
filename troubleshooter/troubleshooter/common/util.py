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

def isfile_check(file_name, name_str):
    if file_name is not None:
        file_name = validate_and_normalize_path(file_name)
        if not os.path.isfile(file_name):
           raise ValueError("The parameter '{0}' must be a file".format(name_str))


def none_and_isfile_check(file_name, name_str):
    if file_name is None:
        raise ValueError("The parameter '{0}' cannot be None.".format(name_str))
    else:
        file_name = validate_and_normalize_path(file_name)
        if not os.path.isfile(file_name):
           raise ValueError("The parameter '{0}' must be a file".format(name_str))

def all_none_or_isfile_check(file_name, file_name_str, obj, obj_str):
    if file_name is not None:
        isfile_check(file_name, file_name_str)
    elif file_name is None and obj is None:
        raise ValueError("The parameters '{0}' or '{1}' must be set to one".format(file_name_str, obj_str))

def dir_exist_check(file_path, name_str):
    if file_path is not None:
        file_dir, _ = os.path.split(validate_and_normalize_path(file_path))
        if not os.path.exists(file_dir):
            raise ValueError("The parameter '{0}' error,"
                             "The dir '{1}' does not exist. Please create the path first.".format(name_str, file_dir))



def none_and_isdir_check(file_dir, name_str):
    if file_dir is None:
        raise ValueError("The parameter '{0}' cannot be None.".format(name_str))
    else:
        if not os.path.isdir(file_dir) or not os.path.exists(file_dir):
            raise ValueError("The parameter '{0}' error,"
                             "'{1}' is not a dir or the dir does not exist.".format(name_str, file_dir))


def type_check(param, name_str, param_type):
    if param and not isinstance(param, param_type):
        raise TypeError("The parameter '{0}' must be bool".format(name_str))

def clear_tmp_file(file):
    if file:
        os.remove(file)

