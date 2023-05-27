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
import mindspore as ms
import torch
import time
from collections import OrderedDict
from pprint import pprint
from troubleshooter.common.format_msg import print_weight_compare_result, print_convert_result ,print_diff_result
from troubleshooter.migrator.mapping_relation.weight_mapping_lib import weight_name_map, weight_value_map
from troubleshooter.migrator.diff_handler import cal_algorithm
from troubleshooter.toolbox.widget_pocket import object_load ,object_dump
from troubleshooter import log as logger
from troubleshooter.common.util import isfile_check, validate_and_normalize_path,\
    none_and_isfile_check, all_none_or_isfile_check, dir_exist_check, type_check, clear_tmp_file

__all__ = [
    "get_weight_map",
    "convert_weight",
    "compare_ms_ckpt",
    "compare_pth_and_ckpt",
]


def _get_para_dict(pth_file_path):
    pt_object = torch.load(pth_file_path, map_location='cpu')
    if isinstance(pt_object, OrderedDict):
        pt_para_dict = pt_object
    elif isinstance(pt_object, torch.nn.Module):
        pt_para_dict = pt_object.state_dict()
    else:
        raise ValueError("PTH file parsing failed, possible reasons: "
                         "1) If using a custom method to save parameter files, please load and set "
                         "the 'pth_para_dict' parameter yourself to use the conversion tool."
                         "2) If the input is an optimizer parameter, this tool does not support "
                         "the conversion of optimizer parameters.")

    values = list(pt_para_dict.values())
    if values and not isinstance(values[0], torch.Tensor):
        raise ValueError("PTH file parsing failed, possible reasons: "
                         "1) If using a custom method to save parameter files, please load and set "
                         "the 'pth_para_dict' parameter yourself to use the conversion tool."
                         "2) If the input is an optimizer parameter, this tool does not support "
                         "the conversion of optimizer parameters.")
    return pt_para_dict


def _get_object(name):
    object_res = None
    index = name.rfind(".")
    if index:
        module_name = name[:index]
        class_name = name[index + 1:]
        import importlib
        imp_module = importlib.import_module(module_name)
        object_res = getattr(imp_module, class_name)
    return object_res


def _get_trans_map(weight_name, module, weight_map, igone_name=False):
    res_weight_map = {}
    for api_name in weight_map:
        obj = _get_object(api_name)
        if isinstance(module, obj):
            para_map = weight_map.get(api_name)
            for pt_para_name, ms_para_name in para_map.items():
                pt_para_item = weight_name + "." + pt_para_name
                if igone_name:
                    ms_para_item = ms_para_name
                else:
                    ms_para_item = weight_name + "." + ms_para_name
                res_weight_map[pt_para_item] = ms_para_item
            break

    return res_weight_map


def _custorm_weight_name_prefix(weight_name_map, prefix=None):
    if prefix:
        custorm_name_map = {}
        for key, value in weight_name_map.items():
            # print(key, ":", prefix + '.' + value)
            custorm_name_map[key] = str(prefix) + '.' + str(value)
        return custorm_name_map
    else:
        return weight_name_map


def get_weight_map(pt_model=None, weight_map_save_path=None, weight_name_prefix=None, custom_name_func=None, print_map=False, **kwargs):
    res_weight_name_map = {}
    res_weight_value_map = {}
    full_weight_name_map = {}

    if not isinstance(pt_model, torch.nn.Module):
        raise TypeError("The parameter 'pt_model' must be torch.nn.Module")

    if custom_name_func and not callable(custom_name_func):
        raise TypeError("The parameter 'custom_name_func' must be a function")

    dir_exist_check(weight_map_save_path, 'weight_map_save_path')
    type_check(weight_name_prefix, 'weight_name_prefix', str)
    type_check(print_map, 'print_map', bool)

    for name, module in pt_model.named_modules():
        tmp_name_map = _get_trans_map(name, module, weight_name_map)
        if tmp_name_map:
            res_weight_name_map.update(tmp_name_map)

        tmp_value_map = _get_trans_map(name, module, weight_value_map, igone_name=True)
        if tmp_value_map:
            res_weight_value_map.update(tmp_value_map)

    for key, value in pt_model.state_dict().items():
        full_weight_name_map[key] = key

    full_weight_name_map.update(res_weight_name_map)
    res_weight_name_map = full_weight_name_map
    res_weight_name_map = _custorm_weight_name_prefix(res_weight_name_map, weight_name_prefix)

    if custom_name_func:
        res_weight_name_map = custom_name_func(res_weight_name_map)

    if print_map:
        pprint(res_weight_name_map)
        pprint(res_weight_value_map)
    if weight_map_save_path:
        object_dump([res_weight_name_map,res_weight_value_map], weight_map_save_path)
    return res_weight_name_map, res_weight_value_map

weight_map_dump=object_dump

def convert_weight(weight_map_path=None, pt_file_path=None, ms_file_save_path=None, **kwargs):
    print_params_list = []

    print_conv_info = kwargs.get('print_conv_info', True)
    pt_param_dict = kwargs.get('pt_param_dict', None)
    weight_map = kwargs.get('weight_map', None)

    all_none_or_isfile_check(weight_map_path,'weight_map_path',weight_map ,'weight_map')
    all_none_or_isfile_check(pt_file_path, 'pt_file_path', pt_param_dict, 'pt_param_dict')
    dir_exist_check(ms_file_save_path,'ms_file_save_path')
    type_check(print_conv_info,'print_conv_info', bool)

    if weight_map:
        name_map, value_map = weight_map
    else:
        name_map, value_map = object_load(weight_map_path)

    if pt_param_dict:
        pt_param_dict = pt_param_dict
    else:
        pt_param_dict = _get_para_dict(pt_file_path)

    new_params_list = []

    for pth_param_name in pt_param_dict:
        # get ckpt name and value
        parameter = pt_param_dict[pth_param_name]
        ms_tensor = ms.Tensor(parameter.numpy())
        # Update name based on name mapping
        ms_para_item = name_map.get(pth_param_name)

        # Update values based on parameter value mapping
        if value_map is not None:
            func = value_map.get(pth_param_name)
        if func:
            def_get_value = _get_object(func)
            ms_tensor = def_get_value(ms_tensor)
        print_params_list.append((pth_param_name, ms_para_item, bool(ms_para_item!=pth_param_name),
                                  bool(func), parameter.size(), ms_tensor.shape))
        # add name and value to list
        new_params_list.append({"name": ms_para_item, "data": ms_tensor})

    if new_params_list is None:
        logger.user_warning("There are no parameters to be converted. Parameter conversion failed. "
                            "Please check whether the configuration is correct")
    if ms_file_save_path:
        ms.save_checkpoint(new_params_list, ms_file_save_path)

    if print_conv_info:
        print_convert_result(print_params_list)
    logger.user_attention("The PTH has been converted to the checkpoint of MindSpore. "
                          "Please check whether the conversion result is correct. "
                          "The saved path is: %s", ms_file_save_path)


def compare_ms_ckpt(orig_file_path=None, target_file_path=None, **kwargs):
    name_map_list = []
    value_map_list = []
    name_map = None
    title = 'The list of comparison results for values'
    field_names = ["Parameter name of original ckpt", "Parameter name of target ckpt",
                   "results of comparison", "match ratio",
                   "cosine similarity", "(mean, max, min)"]
    field_names_pth = ["Parameter name of torch", "Parameter name of MindSpore",
                   "results of comparison", "match ratio",
                   "cosine similarity", "(mean, max, min)"]
    field_names_shape_pth=["Parameter name of torch", "Parameter name of MindSpore",
                           "Whether shape are equal", "Parameter shape of torch",
                           "Parameter shape of MindSpore"]

    compare_value = kwargs.get('compare_value', False)
    weight_map_path = kwargs.get('weight_map_path', None)
    weight_map = kwargs.get('weight_map', None)
    orig_ckpt_dict = kwargs.get('orig_ckpt_dict', None)
    target_ckpt_dict = kwargs.get('target_ckpt_dict', None)
    print_level = kwargs.get('print_level', 1)
    rtol = kwargs.get('rtol', 1e-04)
    atol = kwargs.get('atol', 1e-08)
    equal_nan = kwargs.get('equal_nan', False)

    all_none_or_isfile_check(orig_file_path, 'orig_file_path', orig_ckpt_dict, 'orig_ckpt_dict')
    all_none_or_isfile_check(target_file_path, 'target_file_path', target_ckpt_dict, 'target_ckpt_dict')
    type_check(compare_value,'compare_value', bool)
    type_check(print_level, 'print_level', int)
    isfile_check(weight_map_path,'weight_map_path')

    if orig_file_path:
        orig_ckpt_dict = ms.load_checkpoint(orig_file_path)
    if target_file_path:
        target_ckpt_dict = ms.load_checkpoint(target_file_path)

    if weight_map_path:
        name_map, _ = object_load(weight_map_path)
        name_map =  dict(zip(name_map.values(), name_map.keys()))

    if weight_map:
        name_map, _ = weight_map
        name_map = dict(zip(name_map.values(), name_map.keys()))

    for orig_name, orig_parameter in orig_ckpt_dict.items():
        if name_map:
            new_orig_name = name_map.get(orig_name)
        else:
            new_orig_name = orig_name

        target_para = target_ckpt_dict.get(orig_name)
        target_para_name = orig_name

        if target_para is not None:
            name_map_list.append((new_orig_name, target_para_name, (target_para.shape == orig_parameter.shape),
                                  orig_parameter.shape, target_para.shape))
            if compare_value:
                result, rel_ratio, cosine_sim, diff_detail =cal_algorithm(orig_parameter.value().asnumpy(),
                                                                          target_para.value().asnumpy(),
                                                                          rtol,
                                                                          atol,
                                                                          equal_nan)
                value_map_list.append((new_orig_name, target_para_name, result, rel_ratio, cosine_sim, diff_detail))
            target_ckpt_dict.pop(target_para_name)
        else:
            name_map_list.append((new_orig_name, None, None, orig_parameter.shape, None))

    for name, target_para in target_ckpt_dict.items():
        name_map_list.append((None, name, None, None, target_para.shape))

    if weight_map_path:
        print_weight_compare_result(name_map_list, print_type=print_level, field_names=field_names_shape_pth)
        print_diff_result(value_map_list, title, field_names_pth)
    else:
        print_weight_compare_result(name_map_list, print_type=print_level)
        print_diff_result(value_map_list, title, field_names)


def compare_pth_and_ckpt(weight_map_path=None, pt_file_path=None, ms_file_path=None, **kwargs):
    temp_save_path = validate_and_normalize_path(kwargs.get('temp_save_path', './'))
    timestamp = time.time()
    formatted_time = time.strftime('%Y%m%d-%H%M%S', time.localtime(timestamp))
    temp_name = formatted_time + "_ts_temp_file.ckpt"
    tmp_ckpt_file = os.path.join(temp_save_path, temp_name)
    print_level = kwargs.get('print_level', 1)
    print_conv_info = kwargs.get('print_conv_info', False)


    none_and_isfile_check(weight_map_path, 'weight_map_path')
    none_and_isfile_check(pt_file_path, 'pt_file_path')
    none_and_isfile_check(ms_file_path, 'ms_file_path')

    convert_weight(weight_map_path=weight_map_path, pt_file_path=pt_file_path, ms_file_save_path=tmp_ckpt_file,
                                  print_conv_info=print_conv_info)
    compare_ms_ckpt(orig_file_path=tmp_ckpt_file, target_file_path=ms_file_path,
                    compare_value=True, weight_map_path=weight_map_path, print_level=print_level)
    clear_tmp_file(tmp_ckpt_file)

