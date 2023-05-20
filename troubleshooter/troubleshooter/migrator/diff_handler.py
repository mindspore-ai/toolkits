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
import numpy as np
import os
from troubleshooter.common.format_msg import print_diff_result
from troubleshooter.common.util import validate_and_normalize_path, find_file
from troubleshooter import log as logger


class DifferenceFinder:

    def __init__(self, orig_dir, target_dir):
        self.orig_dir = orig_dir
        self.target_dir = target_dir

    def get_filename_map_list(self):
        name_map_list = []
        orig_name_list = find_file(self.orig_dir)
        orig_name_list.sort()
        target_name_list = find_file(self.target_dir)
        none_flag = False

        if not (orig_name_list and target_name_list):
            logger.user_error("The comparison file is not found in the directory. Please \
                check whether the directory is correct")
            exit(1)

        for name in orig_name_list:
            if name in target_name_list:
                name_map_list.append((name, name))
                target_name_list.remove(name)
            else:
                name_map_list.append((name, None))
                none_flag = True

        if target_name_list:
            target_name_list.sort()
            for name in target_name_list:
                name_map_list.append((None, name))
            none_flag = True

        if none_flag:
            logger.user_warning("The files in the original directory and the target directory cannot be fully mapped. "
                                "Please manually complete the mapping of file names")
            print("filename mapping list:" + str(name_map_list))
        return name_map_list

    def compare_npy_dir(self, name_map_list=None, **kwargs):
        """
        """
        if name_map_list is None:
            name_map_list = self.get_filename_map_list()

        rtol = kwargs.get('rtol', 1e-04)
        atol = kwargs.get('atol', 1e-08)
        equal_nan = kwargs.get('equal_nan', False)

        result_list = []
        normal_orig_dir = validate_and_normalize_path(self.orig_dir)
        normal_target_dir = validate_and_normalize_path(self.target_dir)
        for name_map in name_map_list:
            orig_name = name_map[0]
            target_name = name_map[1]

            if orig_name is None or target_name is None:
                result = False
                diff_detail = ()
                result_list.append(
                    (orig_name, target_name, result, diff_detail))
                continue

            orig_file = os.path.join(normal_orig_dir, orig_name)
            target_file = os.path.join(normal_target_dir, target_name)

            if not os.path.isfile(orig_file) or not os.path.isfile(target_file):
                continue

            orig_value = np.load(orig_file)
            target_value = np.load(target_file)
            result, rel_ratio, cosine_sim, diff_detail = cal_algorithm(orig_value, target_value, rtol, atol, equal_nan)
            result_list.append((orig_name, target_name, result, rel_ratio, cosine_sim, diff_detail))
        logger.user_attention("The compare directory information:\n The orig dir: %s \n The target dir: %s",
                              self.orig_dir, self.target_dir)
        print_diff_result(result_list)


def cal_algorithm(orig_value, target_value, rtol, atol, equal_nan):
    if orig_value.shape == target_value.shape:
        result = np.allclose(orig_value, target_value, rtol=rtol, atol=atol, equal_nan=equal_nan)
        if not result:
            value_diff = np.abs(orig_value - target_value)
            value_mean = value_diff.mean()
            value_max = value_diff.max()
            value_min = value_diff.min()
            diff_detail = value_mean, value_max, value_min
        else:
            diff_detail = ()
        cosine_sim = cal_cosine_sim(orig_value, target_value)
        rel_ratio = np.isclose(orig_value, target_value, rtol=rtol, atol=atol,
                               equal_nan=equal_nan).sum()/np.size(orig_value)
    else:
        result = False
        diff_detail = ("Shape is inconsistent",
                       orig_value.shape, target_value.shape)

    return result, rel_ratio, cosine_sim, diff_detail

def cal_cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    sim = 0.
    num = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if not denom == 0.:
        sim = 0.5 + 0.5 * (num / denom)
    return sim

