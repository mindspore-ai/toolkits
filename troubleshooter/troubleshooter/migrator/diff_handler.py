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
"""compare tools"""
import os
import torch
import numpy as np
from collections import OrderedDict
from pprint import pprint
from troubleshooter import log as logger
from troubleshooter.common.util import validate_and_normalize_path, find_file, make_directory
from troubleshooter.migrator.mapping_relation.weight_mapping_lib import weight_name_map, weight_value_map
from troubleshooter.common.format_msg import print_diff_result, print_weight_compare_result, print_convert_result

FRAMEWORK_TYPE = "ms"

try:
    import mindspore as ms
except ModuleNotFoundError as e:
    e_msg = e.msg
    no_module_msg = "No module named 'mindspore'"
    if e_msg == no_module_msg:
        FRAMEWORK_TYPE = 'pt'
    else:
        raise e

class TensorRecorder:
    def __init__(self):
        self.summary_record = None
        self.count = 1

    def init_tensor_summary(self, record_path=None, record_mode=None):
        if record_mode is None:
            record_mode = ms.get_context("mode")

        if record_mode == 0:
            self.summary_record = ms.SummaryRecord(record_path, export_options={'tensor_format': 'npy'})

    def record(self):
        if not self.summary_record:
            self.summary_record.record()
            self.summary_record.flush()

    @staticmethod
    def record_op(record_dir=None, record_mode=None, framework=None):
        normal_dir = None

        if framework is None:
            framework = FRAMEWORK_TYPE

        if record_dir is not None:
            normal_dir = validate_and_normalize_path(record_dir)
            make_directory(normal_dir)

        if framework=="ms":
            if record_mode is None:
                record_mode = ms.get_context("mode")
            if record_mode == 0:
                if record_dir is not None:
                    logger.user_warning("In MindSpore graph mode, only TensorSummary can be used to record data. "
                                        "In this mode, The parameter 'record_dir' is not valid")
                save_op = ms.ops.TensorSummary()
            else:
                def save_t2n(name, value):
                    if normal_dir is not None:
                        name = os.path.join(normal_dir, name)
                    np.save(name+".npy", value.asnumpy())
                save_op = save_t2n
        else:
            def save_t2n(name, value):
                if normal_dir is not None:
                    name = os.path.join(normal_dir, name)
                np.save(name + ".npy", value.detach().numpy())
            save_op = save_t2n

        return save_op

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
            logger.user_error("The comparison file is not found in the directory. "
                              "Please check whether the directory is correct")
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

        rtol = kwargs.get('rtol', 1e-05)
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
                result_list.append((orig_name, target_name, result, diff_detail))
                continue

            orig_file = os.path.join(normal_orig_dir, orig_name)
            target_file = os.path.join(normal_target_dir, target_name)

            if not os.path.isfile(orig_file) or not os.path.isfile(target_file):
                continue

            orig_value = np.load(orig_file)
            target_value = np.load(target_file)
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
            else:
                result = False
                diff_detail = ("Shape is inconsistent", orig_value.shape, target_value.shape)

            result_list.append((orig_name, target_name, result, diff_detail))
        logger.user_attention("The compare directory information:\n The orig dir: %s \n The target dir: %s",
                              self.orig_dir, self.target_dir)
        print_diff_result(result_list)

class WeightMigrator:

    def __init__(self, pt_model=None, pth_file_path=None, pth_para_dict=None, ckpt_save_path=None):
        self.weight_map = weight_name_map
        self.ckpt_path = ckpt_save_path
        self.pt_model = pt_model
        self.pt_para_dict = self._get_para_dict(pth_file_path, pth_para_dict)
        self.print_params_list = []


    def _get_para_dict(self, pth_file_path, pth_para_dict):
        if pth_para_dict:
            return pth_para_dict

        pt_para_dict = {}
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

    def _get_object(self, name):
        object_res = None
        index = name.rfind(".")
        if index:
            module_name = name[:index]
            class_name = name[index + 1:]
            import importlib
            imp_module = importlib.import_module(module_name)
            object_res = getattr(imp_module, class_name)
        return object_res


    def _get_trans_map(self, weight_name, module, weight_map, igone_name=False):
        res_weight_map = {}
        for api_name in weight_map:
            obj = self._get_object(api_name)
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


    def _custorm_weight_name_prefix(self, weight_name_map, prefix=None):
        if prefix:
            custorm_name_map = {}
            for key, value in weight_name_map.items():
                # print(key, ":", prefix + '.' + value)
                custorm_name_map[key] = str(prefix) + '.' + str(value)
            return custorm_name_map
        else:
            return weight_name_map


    def get_weight_map(self, print_map=False, full_name_map=False):
        res_weight_name_map = {}
        res_weight_value_map = {}
        full_weight_name_map = {}

        for name, module in self.pt_model.named_modules():
            tmp_name_map = self._get_trans_map(name, module, weight_name_map)
            if tmp_name_map:
                res_weight_name_map.update(tmp_name_map)
            tmp_value_map = self._get_trans_map(name, module, weight_value_map, igone_name=True)
            if tmp_value_map:
                res_weight_value_map.update(tmp_value_map)
        if full_name_map:
            for key, value in self.pt_para_dict.items():
                full_weight_name_map[key]=key
            full_weight_name_map.update(res_weight_name_map)
            res_weight_name_map =  full_weight_name_map

        if print_map:
            pprint(res_weight_name_map)
            pprint(res_weight_value_map)
        return res_weight_name_map, res_weight_value_map


    def _get_name_and_value(self, pth_param_name, name_map, value_map):
        new_name = pth_param_name
        parameter = self.pt_para_dict[pth_param_name]
        ms_tensor = ms.Tensor(parameter.numpy())

        # Update name based on name mapping
        ms_para_item = name_map.get(pth_param_name)
        if ms_para_item:
            new_name = ms_para_item

        # Update values based on parameter value mapping
        if value_map is not None:
            fun = value_map.get(pth_param_name)

        if fun:
            def_get_value = self._get_object(fun)
            ms_tensor = def_get_value(ms_tensor)

        self.print_params_list.append((pth_param_name, new_name, bool(ms_para_item), bool(fun) , parameter.size(),
                                       ms_tensor.shape))
        return new_name, ms_tensor


    def convert(self, weight_name_map=None, weight_value_map=None ,weight_name_prefix=None, print_conv_info=True):

        if weight_name_prefix:
            name_map, value_map = self.get_weight_map(full_name_map=True)
            name_map = self._custorm_weight_name_prefix(name_map, weight_name_prefix)
        else:
            name_map, value_map = self.get_weight_map()

        if weight_name_map is not None:
            name_map =  weight_name_map
        if weight_value_map is not None:
            value_map =  weight_value_map

        new_params_list = []

        for pth_param_name in self.pt_para_dict:
            # get ckpt name and value
            new_name, ms_tensor = self._get_name_and_value(pth_param_name,name_map,value_map)
            # add name and value to list
            new_params_list.append({"name": new_name, "data": ms_tensor})

        if new_params_list:
            ms.save_checkpoint(new_params_list , self.ckpt_path)
        else:
            logger.user_warning("There are no parameters to be converted. Parameter conversion failed. "
                                "Please check whether the configuration is correct")

        if print_conv_info:
             print_convert_result(self.print_params_list)

        logger.user_attention("The PTH has been converted to the checkpoint of MindSpore. "
                              "Please check whether the conversion result is correct. "
                              "The saved path is: %s",self.ckpt_path)


    def compare_ckpt(self, ckpt_path=None, converted_ckpt_path=None, print_result=1):
        name_map_list = []
        if converted_ckpt_path is None:
            ckpt_after_convert_path = self.ckpt_path
        ckpt_dict = ms.load_checkpoint(ckpt_path)
        ckpt_after_conv_dict = ms.load_checkpoint(ckpt_after_convert_path)

        for ms_para_name, ms_para in ckpt_dict.items():
            ms_para_after_conv = ckpt_after_conv_dict.get(ms_para_name)

            if ms_para_after_conv is not None:
                name_map_list.append((ms_para_name, ms_para_name, (ms_para.shape == ms_para_after_conv.shape),
                                      ms_para.shape, ms_para_after_conv.shape))
                ckpt_after_conv_dict.pop(ms_para_name)
            else:
                name_map_list.append((ms_para_name, None, None, ms_para.shape, None))

        for name, ms_para in ckpt_after_conv_dict.items():
            name_map_list.append((None, name, None, None, ms_para.shape))
        print_weight_compare_result(name_map_list, print_type=print_result)
