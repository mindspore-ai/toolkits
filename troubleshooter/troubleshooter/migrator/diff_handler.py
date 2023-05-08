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
import csv
import time
from pprint import pprint
import torch
import numpy as np
from troubleshooter import log as logger
from troubleshooter.common.util import validate_and_normalize_path, find_file, make_directory, \
    cal_similarity, cal_cosine_sim, save_numpy_data
from troubleshooter.migrator.mapping_relation.weight_mapping_lib import weight_name_map, weight_value_map
from troubleshooter.common.format_msg import print_diff_result, print_weight_compare_result, \
    print_convert_result, print_net_infer_diff_result

FRAMEWORK_TYPE = "ms"
MS_OUTPUT_PATH = "data/output/MindSpore"
PT_OUTPUT_PATH = "data/output/PyTorch"
RESULT_COLUMNS = ["Pytorch data", "MindSpore data",
                  "Results of comparison", "cosine similarity", "(mean, max, min)"]

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
            self.summary_record = ms.SummaryRecord(
                record_path, export_options={'tensor_format': 'npy'})

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

        if framework == "ms":
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
                result_list.append(
                    (orig_name, target_name, result, diff_detail))
                continue

            orig_file = os.path.join(normal_orig_dir, orig_name)
            target_file = os.path.join(normal_target_dir, target_name)

            if not os.path.isfile(orig_file) or not os.path.isfile(target_file):
                continue

            orig_value = np.load(orig_file)
            target_value = np.load(target_file)
            if orig_value.shape == target_value.shape:
                result = np.allclose(
                    orig_value, target_value, rtol=rtol, atol=atol, equal_nan=equal_nan)

                if not result:
                    value_diff = np.abs(orig_value - target_value)
                    value_mean = value_diff.mean()
                    value_max = value_diff.max()
                    value_min = value_diff.min()
                    cosine_sim = cal_cosine_sim(orig_value, target_value)
                    diff_detail = value_mean, value_max, value_min
                else:
                    diff_detail = ()
                    cosine_sim = cal_cosine_sim(orig_value, target_value)
            else:
                result = False
                diff_detail = ("Shape is inconsistent",
                               orig_value.shape, target_value.shape)

            result_list.append(
                (orig_name, target_name, result, cosine_sim, diff_detail))
        logger.user_attention("The compare directory information:\n The orig dir: %s \n The target dir: %s",
                              self.orig_dir, self.target_dir)
        print_diff_result(result_list)


class WeightMigrator:

    def __init__(self, pt_model=None, pth_file_path=None, ckpt_save_path=None):
        self.weight_map = weight_name_map
        self.ckpt_path = ckpt_save_path
        self.pt_model = pt_model
        self.pt_para_dict = torch.load(pth_file_path, map_location='cpu')
        self.print_params_list = []

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

    def get_weight_map(self, print_map=False):
        res_weight_name_map = {}
        res_weight_value_map = {}
        for name, module in self.pt_model.named_modules():
            tmp_name_map = self._get_trans_map(name, module, weight_name_map)
            if tmp_name_map:
                res_weight_name_map.update(tmp_name_map)
            tmp_value_map = self._get_trans_map(
                name, module, weight_value_map, igone_name=True)
            if tmp_value_map:
                res_weight_value_map.update(tmp_value_map)
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

        self.print_params_list.append((pth_param_name, new_name, bool(ms_para_item), bool(fun), parameter.size(),
                                       ms_tensor.shape))
        return new_name, ms_tensor

    def convert(self, weight_name_map=None, weight_value_map=None, print_conv_info=True):
        name_map, value_map = self.get_weight_map()
        if weight_name_map is not None:
            name_map = weight_name_map
        if weight_value_map is not None:
            value_map = weight_value_map

        new_params_list = []

        for pth_param_name in self.pt_para_dict:
            # get ckpt name and value
            new_name, ms_tensor = self._get_name_and_value(
                pth_param_name, name_map, value_map)
            # add name and value to list
            new_params_list.append({"name": new_name, "data": ms_tensor})

        if new_params_list:
            ms.save_checkpoint(new_params_list, self.ckpt_path)
        else:
            logger.user_warning("There are no parameters to be converted. Parameter conversion failed. "
                                "Please check whether the configuration is correct")

        if print_conv_info:
            print_convert_result(self.print_params_list)

        logger.user_attention("The PTH has been converted to the checkpoint of MindSpore. "
                              "Please check whether the conversion result is correct. "
                              "The saved path is: %s", self.ckpt_path)

    def compare_ckpt(self, ckpt_path=None, converted_ckpt_path=None, print_result=1):
        name_map_list = []
        if converted_ckpt_path is None:
            ckpt_after_convert_path = self.ckpt_path
        ckpt_dict = ms.load_checkpoint(ckpt_path)
        ckpt_after_conv_dict = ms.load_checkpoint(ckpt_after_convert_path)

        for ms_para_name, ms_para in ckpt_dict.items():
            ms_para_after_conv = ckpt_after_conv_dict.get(ms_para_name)

            if ms_para_after_conv is not None:
                name_map_list.append((ms_para_name, ms_para_name, (ms_para.shape ==
                                     ms_para_after_conv.shape), ms_para.shape, ms_para_after_conv.shape))
                ckpt_after_conv_dict.pop(ms_para_name)
            else:
                name_map_list.append(
                    (ms_para_name, None, None, ms_para.shape, None))

        for name, ms_para in ckpt_after_conv_dict.items():
            name_map_list.append((None, name, None, None, ms_para.shape))
        print_weight_compare_result(name_map_list, print_type=print_result)


class NetDifferenceFinder:

    def __init__(self, ms_net, pt_net, inputs,
                 out_path, print_result):
        self.ms_net = ms_net
        self.pt_net = pt_net
        self.inputs = inputs
        self.out_path = out_path
        self.print_result = print_result

    def start_compare(self):
        compare_results = []
        for idx, input in enumerate(self.inputs):
            input_data = self.get_input_data(input)
            result_ms, result_pt = self.infer_net(
                input_data, self.ms_net, self.pt_net, idx)
            self.check_output(result_ms, result_pt)
            if idx != 0:
                compare_results.append(['', '', '', '', ''])
            compare_results.extend(
                self.compare_results(result_ms, result_pt, idx))
        self.save_results(compare_results)
        print_net_infer_diff_result(compare_results)

    def get_input_data(self, input):
        input_data = []
        if isinstance(input, dict):
            input_data = list(input.values())
        else:
            for data in input:
                if isinstance(data, str):
                    input_data.append(np.load(data))
                elif isinstance(data, np.ndarray):
                    input_data.append(data)
                else:
                    logger.user_error(
                        'Unknow input data type {}'.format(type(data)))
                    exit(1)
        return input_data

    def infer_net(self, input_data, ms_net, pt_net, idx):
        if self.print_result:
            print(
                "\n=================================Start inference net=================================")
        start_pt = time.time()
        result_pt = self.run_pt_net(pt_net, input_data)
        end_pt = time.time()
        print(f"In test case {idx}, the PyTorch net inference completed cost %.5f seconds." % (
            end_pt - start_pt))
        start_ms = time.time()
        result_ms = self.run_ms_net(ms_net, input_data)
        end_ms = time.time()
        print(f"In test case {idx}, the MindSpore net inference completed cost %.5f seconds." % (
            end_ms - start_ms))
        if isinstance(result_ms, tuple):
            result_ms = {f"result_{idx}": result for idx,
                         result in enumerate(result_ms)}
        if isinstance(result_pt, tuple):
            result_pt = {f"result_{idx}": result for idx,
                         result in enumerate(result_pt)}
        return result_ms, result_pt

    def run_pt_net(self, pt_net, input_data_list):
        data_list = []
        for data in input_data_list:
            data_list.append(torch.tensor(data))
        pt_results = pt_net(*data_list)
        return pt_results

    def run_ms_net(self, ms_net, input_data_list):
        data_list = []
        for data in input_data_list:
            data_list.append(ms.Tensor(data))
        ms_results = ms_net(*data_list)
        return ms_results

    def check_output(self, result_ms, result_pt):
        ms_result_num = len(result_ms)
        if self.print_result:
            print("The MindSpore net inference have %s result." % ms_result_num)
        pt_result_num = len(result_pt)
        if self.print_result:
            print("The PyTorch net inference have %s result." % pt_result_num)
        assert ms_result_num == pt_result_num, "output results are in different counts!"

    def compare_results(self, result_ms, result_pt, idx):
        index = 0
        compare_result = []
        for (k_pt, k_ms) in zip(result_pt, result_ms):
            result_pt_ = result_pt[k_pt].detach().numpy()
            result_ms_ = result_ms[k_ms].asnumpy()
            self.check_out_data_shape(index, result_ms_, result_pt_)
            self.save_out_data(index, k_ms, k_pt, result_ms_, result_pt_)
            result_pt_ = result_pt_.reshape(1, -1)
            result_ms_ = result_ms_.reshape(1, -1)
            result = self.compare_data(result_ms_, result_pt_, index)
            result[0], result[1] = f"test{idx}-{k_ms}", f"test{idx}-{k_pt}"
            result[3] = "%.5f" % float(result[3])
            min_max = ['%.5f' % r for r in result[4]]
            result[4] = min_max
            compare_result.append(result)
            index += 1
        return compare_result

    def check_out_data_shape(self, index, result_ms, result_pt):
        if self.print_result:
            print(
                "\n=========================Start Check Out Data %s ===========================" % index)
        pt_out_shape = result_pt.shape
        ms_out_shape = result_pt.shape
        assert pt_out_shape == ms_out_shape, "output results are in different shapes!"
        if self.print_result:
            print("shape of result_pt: %s" % str(pt_out_shape))
            print("shape of result_ms: %s" % str(ms_out_shape))
            print("-result_pt-: \n", result_pt)
            print("-result_ms-: \n", result_ms)

    def save_out_data(self, index, k_ms, k_pt, result_ms, result_pt):
        if self.print_result:
            print(
                "\n================= ======Start Save Out Data %s =========================" % index)
        result_file = "%s.npy" % k_ms
        ms_out_path = os.path.join(self.out_path, MS_OUTPUT_PATH, result_file)
        save_numpy_data(ms_out_path, result_ms)
        if self.print_result:
            print("Saved MindSpore output data at: %s" % ms_out_path)

        result_file = "%s.npy" % k_pt
        pt_out_path = os.path.join(self.out_path, PT_OUTPUT_PATH, result_file)
        save_numpy_data(pt_out_path, result_pt)
        if self.print_result:
            print("Saved PyTorch output data at: %s" % pt_out_path)

    def compare_data(self, result_ms, result_pt, index):
        if self.print_result:
            print(
                "\n=========================Start Compare Out Data %s ===========================" % index)
        sim_result = cal_similarity(result_ms, result_pt, index)
        return sim_result

    def save_results(self, compare_results):
        if self.print_result:
            logger.info(
                "=================================Start save result=================================")
        result_path = os.path.join(self.out_path, "compare_result.csv")
        with open(result_path, "w", encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(RESULT_COLUMNS)
            writer.writerows(compare_results)
            logger.info(
                "The comparison result have been written to %s" % result_path)
