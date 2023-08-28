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
"""compare tools"""
import functools
import math
import multiprocessing
import os
from itertools import zip_longest

import numpy as np
from tqdm import tqdm

from troubleshooter import log as logger
from troubleshooter.common.format_msg import print_diff_result
from troubleshooter.common.util import (find_file, none_and_isdir_check, type_check,
                                        validate_and_normalize_path)

__all__ = [
    "get_name_map_list_by_name",
    "get_name_map_list_by_number",
    "get_name_map_list_by_shape_edit_distance",
    "compare_npy_dir",
    "compare_list_dir",
    "compare_dict_dir",
    "compare_grads_dir",
    "cal_algorithm",
    "cal_cosine_sim"
]

def get_name_map_list_by_name(orig_dir, target_dir):
    name_map_list = []
    orig_name_list = find_file(orig_dir)
    target_name_list = find_file(target_dir)
    none_flag = False

    if not (orig_name_list and target_name_list):
        raise ValueError("The comparison file is not found in the directory. Please \
            check whether the directory is correct")

    for name in orig_name_list:
        if name in target_name_list:
            name_map_list.append((name, name))
            target_name_list.remove(name)
        else:
            name_map_list.append((name, None))
            none_flag = True

    if target_name_list:
        for name in target_name_list:
            name_map_list.append((None, name))
        none_flag = True

    if none_flag:
        logger.user_warning("The files in the original directory and the target directory cannot be fully mapped. "
                            "Please manually complete the mapping of file names")
    return name_map_list


def get_name_map_list_by_number(orig_dir, target_dir):
    orig_name_list = find_file(orig_dir)
    target_name_list = find_file(target_dir)
    if not (orig_name_list and target_name_list):
        raise ValueError("The comparison file is not found in the directory. Please \
            check whether the directory is correct")
    if len(orig_name_list) != len(target_name_list):
        logger.user_warning("The number of files is not equal. Some files can not be mapped. "
                            "Number of files in the original directory is %d, "
                            "Number of files in the target directory is %d",
                            len(orig_name_list), len(target_name_list))
    return list(zip_longest(orig_name_list, target_name_list, fillvalue=None))


def min_edit_distance(A, B, del_cost=1, ins_cost=1, rep_cost=5):
    m = len(A)
    n = len(B)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    operations = [[''] * (n + 1) for _ in range(m + 1)]
    # Replace cost > (Delete or Insert)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                if dp[i - 1][j] + del_cost == dp[i][j]:
                    operations[i][j] = 'Delete'
                else:
                    operations[i][j] = 'Match'
            else:
                delete_cost = dp[i - 1][j] + del_cost
                insert_cost = dp[i][j - 1] + ins_cost
                replace_cost = dp[i - 1][j - 1] + rep_cost

                if delete_cost <= insert_cost and delete_cost <= replace_cost:
                    dp[i][j] = delete_cost
                    operations[i][j] = 'Delete'
                elif insert_cost <= delete_cost and insert_cost <= replace_cost:
                    dp[i][j] = insert_cost
                    operations[i][j] = 'Insert'
                else:
                    dp[i][j] = replace_cost
                    operations[i][j] = 'Replace'

    i = m
    j = n
    matches = []
    while i > 0 and j > 0:
        if operations[i][j] == 'Match':
            matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif operations[i][j] == 'Delete':
            matches.append((i - 1, None))
            i -= 1
        elif operations[i][j] == 'Insert':
            matches.append((None, j - 1))
            j -= 1
        else:
            matches.append((i - 1, j - 1))
            i -= 1
            j -= 1

    while i > 0:
        matches.append((i - 1, None))
        i -= 1
    while j > 0:
        matches.append((None, j - 1))
        j -= 1

    matches.reverse()
    return dp[m][n], matches


def get_name_map_list_by_shape_edit_distance(orig_dir, target_dir, *, del_cost=1, ins_cost=1, rep_cost=5):
    orig_name_list = find_file(orig_dir)
    target_name_list = find_file(target_dir)
    if not (orig_name_list and target_name_list):
        raise ValueError("The comparison file is not found in the directory. Please \
            check whether the directory is correct")

    orig_shape_list = [np.load(os.path.join(orig_dir, orig_name)).shape for orig_name in orig_name_list]
    target_shape_list = [np.load(os.path.join(target_dir, target_name)).shape for target_name in target_name_list]
    min_cost, index_map_list = min_edit_distance(orig_shape_list, target_shape_list, del_cost, ins_cost, rep_cost)
    if min_cost != 0:
        logger.user_warning("The number of files or their corresponding shapes are inconsistent, "
                            "and some files may not be correctly mapped.")
    name_map_list = [(orig_name_list[a] if a is not None else None, target_name_list[b] if b is not None else None) for
                     a, b in index_map_list]
    return name_map_list


def _cal_compare_npy_single_process(name, normal_orig_dir, normal_target_dir, rtol, atol, equal_nan, compare_shape):
    def load_value(dir, name):
        if name is None:
            return None
        file = os.path.join(dir, name)
        if not os.path.isfile(file):
            return None
        return np.load(file)
    orig_name, target_name = name
    orig_value = load_value(normal_orig_dir, orig_name)
    target_value = load_value(normal_target_dir, target_name)
    result, rel_ratio, cosine_sim, diff_detail = cal_algorithm(
        orig_value, target_value, rtol, atol, equal_nan)
    if compare_shape:
        orig_shape = orig_value.shape if orig_value is not None else None
        target_shape = target_value.shape if target_value is not None else None
        return (orig_name, target_name, orig_shape, target_shape, result, rel_ratio, cosine_sim, diff_detail)
    else:
        return (orig_name, target_name, result, rel_ratio, cosine_sim, diff_detail)


def compare_npy_dir(
    orig_dir,
    target_dir,
    rtol=1e-4,
    atol=1e-4,
    equal_nan=False,
    *,
    name_map_list=None,
    compare_shape=False,
    output_file=None,
    **kwargs
):
    none_and_isdir_check(orig_dir, 'orig_dir')
    none_and_isdir_check(target_dir, 'target_dir')
    type_check(rtol, 'rtol', float)
    type_check(atol, 'atol', float)
    type_check(equal_nan, 'atol', bool)
    type_check(compare_shape, 'compare_shape', bool)
    expected_kwargs = {
        'title': '',
        'field_names': [],
    }

    for key, value in kwargs.items():
        if key in expected_kwargs:
            expected_type = type(expected_kwargs[key])
            if not isinstance(value, expected_type):
                raise TypeError(f"Invalid type for '{key}'. Expected {expected_type}, but got {type(value)}.")
        else:
            raise ValueError(f"Unexpected keyword argument: '{key}'")

    title = kwargs.get('title', None)
    field_names = kwargs.get('field_names', None)

    if name_map_list is None:
        name_map_list = get_name_map_list_by_name(orig_dir, target_dir)

    normal_orig_dir = validate_and_normalize_path(orig_dir)
    normal_target_dir = validate_and_normalize_path(target_dir)
    logger.user_attention(
        "The compare directory information:\n The orig dir: %s \n The target dir: %s",
        orig_dir,
        target_dir,
    )
    with multiprocessing.Pool() as pool:
        _compare_npy_single_process = functools.partial(
            _cal_compare_npy_single_process,
            normal_orig_dir=normal_orig_dir,
            normal_target_dir=normal_target_dir,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            compare_shape=compare_shape,
        )
        result_list = list(tqdm(pool.imap(_compare_npy_single_process, name_map_list), total=len(name_map_list)))

    return print_diff_result(result_list, title=title, field_names=field_names,
                             output_file=output_file, show_shape_diff=compare_shape)


def compare_grads_dir(orig_dir, target_dir, rtol=1e-4, atol=1e-4, equal_nan=False, compare_shape=True, output_file=None):
    none_and_isdir_check(orig_dir, 'orig_dir')
    none_and_isdir_check(target_dir, 'target_dir')
    name_map_list = get_name_map_list_by_shape_edit_distance(orig_dir, target_dir)
    return compare_npy_dir(orig_dir, target_dir, rtol, atol, equal_nan,
                           name_map_list=name_map_list, compare_shape=compare_shape, output_file=output_file)


def compare_list_dir(orig_dir, target_dir, rtol=1e-4, atol=1e-4, equal_nan=False, compare_shape=False, output_file=None):
    none_and_isdir_check(orig_dir, 'orig_dir')
    none_and_isdir_check(target_dir, 'target_dir')
    name_map_list = get_name_map_list_by_number(orig_dir, target_dir)
    return compare_npy_dir(orig_dir, target_dir, rtol, atol, equal_nan,
                           name_map_list=name_map_list, compare_shape=compare_shape, output_file=output_file)


def compare_dict_dir(orig_dir, target_dir, rtol=1e-4, atol=1e-4, equal_nan=False, compare_shape=False, output_file=None):
    none_and_isdir_check(orig_dir, 'orig_dir')
    none_and_isdir_check(target_dir, 'target_dir')
    name_map_list = get_name_map_list_by_name(orig_dir, target_dir)
    return compare_npy_dir(orig_dir, target_dir, rtol, atol, equal_nan,
                           name_map_list=name_map_list, compare_shape=compare_shape, output_file=output_file)


def cal_algorithm(orig_value, target_value, rtol, atol, equal_nan):
    rel_ratio = 0
    cosine_sim = math.nan
    diff_detail = (math.nan, math.nan)

    def handle_dtype(input):
        if np.issubdtype(input.dtype, np.floating):
            return input
        else:
            return input.astype(float)

    if orig_value is None or target_value is None:
        allclose_result = False
        return allclose_result, rel_ratio, cosine_sim, diff_detail

    if orig_value.shape == target_value.shape:
        orig_value = handle_dtype(orig_value)
        target_value = handle_dtype(target_value)

        # For scalar
        if not orig_value.shape:
            value_diff = np.abs(orig_value - target_value) if orig_value != target_value else np.array([])
        else:
            unequal_indices = np.where(orig_value != target_value)
            value_diff = np.abs(orig_value[unequal_indices] - target_value[unequal_indices])

        if value_diff.size > 0:
            value_mean = value_diff.mean()
            value_max = value_diff.max()
            diff_detail = value_mean, value_max
            cosine_sim = cal_cosine_sim(orig_value, target_value)
            isclose_num = np.isclose(orig_value, target_value, rtol=rtol, atol=atol, equal_nan=equal_nan).sum()
            rel_ratio = isclose_num / np.size(orig_value)
            allclose_result = isclose_num == np.size(orig_value)
        else:
            diff_detail = (0., 0.)
            cosine_sim = 1.
            rel_ratio = 1.
            allclose_result = True
    else:
        allclose_result = "Shape is inconsistent"

    return allclose_result, rel_ratio, cosine_sim, diff_detail


def cal_cosine_sim(a, b):
    np.seterr(divide='ignore', invalid='ignore')
    a, b = a.flatten(), b.flatten()
    eps = np.finfo(float).eps

    num = np.dot(a, b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm <= eps and b_norm <= eps:
        result = np.array(1.)
    elif a_norm <= eps or b_norm <= eps:
        result = np.array(np.nan)
    else:
        result = num / (a_norm * b_norm)

    result = float(result.clip(-1, 1))
    return result
