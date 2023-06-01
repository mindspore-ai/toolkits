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
import csv
import mindspore as ms
import numpy as np
import os
import random
import time
import torch
from troubleshooter.migrator import weight_migrator
from troubleshooter import log as logger
from troubleshooter.migrator.diff_handler import cal_algorithm
from troubleshooter.common.util import save_numpy_data, validate_and_normalize_path
from troubleshooter.common.format_msg import print_diff_result

MS_OUTPUT_PATH = "data/output/MindSpore"
PT_OUTPUT_PATH = "data/output/PyTorch"
TITLE = 'The comparison results of Net'
FIELD_NAMES = ["pt data", "ms data",
               "results of comparison", "match ratio", "cosine similarity", "(mean, max, min)"]
__all__ = [
    "NetDifferenceFinder",
]

class NetDifferenceFinder:

    def __init__(self, 
                 pt_net, 
                 ms_net, 
                 print_result=True,
                 pt_path=None,
                 ms_path=None,
                 auto_conv_ckpt=True,
                 fix_random_seed=16,
                 **kwargs):
        self.ms_net = ms_net
        self.pt_net = pt_net
        self.print_result = print_result
        self.pt_path = pt_path
        self.ms_path = ms_path
        self.auto_conv_ckpt = False if pt_path and ms_path else auto_conv_ckpt
        self.fix_random_seed = fix_random_seed
        self.out_path = validate_and_normalize_path(
            kwargs.get('out_path', './'))
        self.pt_org_pth = os.path.join(
            self.out_path, 
            kwargs.get('pt_org_pth', 'net_diff_finder_pt_org_pth.pth'))
        self.conv_ckpt_name = os.path.join(
            self.out_path, 
            kwargs.get('conv_ckpt_name', 'net_diff_finder_conv_ckpt.ckpt'))
        self.ms_org_ckpt = os.path.join(
            self.out_path, 
            kwargs.get('ms_org_ckpt', 'net_diff_finder_ms_org_ckpt.ckpt'))
        
    def _check_input(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise TypeError("The type of inputs must be list or tuple")
        if isinstance(inputs, tuple):
            for item in inputs:
                if not isinstance(item, (ms.Tensor, np.ndarray, torch.Tensor, str)):
                    raise TypeError("The type of inputs must be ms.Tensor, np.ndarray, torch.Tensor or str")
        elif isinstance(inputs, list):
            for items in inputs:
                if not isinstance(items, tuple):
                    raise TypeError("The type of each input must be tuple")
                for item in items:
                    if not isinstance(item, (ms.Tensor, np.ndarray, torch.Tensor, str)):
                        raise TypeError("The type of model input must be ms.Tensor, np.ndarray, torch.Tensor or str")
            
    def _check_auto_input(self, auto_inputs):
        if auto_inputs:
            if isinstance(auto_inputs, tuple):
               for item in auto_inputs:
                   if not isinstance(item, tuple):
                       raise TypeError("The shape of input data must be tuple")
            elif isinstance(auto_inputs, dict):
                if 'input' not in auto_inputs.keys():
                    raise KeyError("The keys of auto_input must contain 'input'")
                if 'num' not in auto_inputs.keys():
                    raise KeyError("The keys of auto_input must contain 'num'")
                for item in auto_inputs['input']:
                    if not isinstance(item, tuple):
                        raise TypeError("The shape of input data must be tuple")
            else:
                raise TypeError("The type of auto_input must be tuple or dict")
            
    def _build_auto_input(self, auto_inputs):
        inputs_ = list()
        inputs = []
        if auto_inputs:
            if isinstance(auto_inputs, tuple):
                for item in auto_inputs:
                    inputs_.append(np.random.randn(*item[0]).astype(item[1]))
                inputs = tuple(inputs_)
            elif isinstance(auto_inputs, dict):
                for _ in range(auto_inputs["num"]):
                    tmp = []
                    for item in auto_inputs['input']:
                        tmp.append(np.random.randn(*item[0]).astype(item[1]))
                    tmp = tuple(tmp)
                inputs.append(tmp)
        return inputs
    
    def _convert_data_format(self, inputs):
        inputs_ = []
        if isinstance(inputs, tuple):
            inputs_.append(inputs)
        elif isinstance(inputs, list):
            inputs_ = inputs
        return inputs_
        
    def compare(self, 
                inputs=None, 
                auto_inputs=None, 
                rtol=1e-04,
                atol=1e-08,
                equal_nan=False):
        self._check_auto_input(auto_inputs)
        if auto_inputs:
            inputs = self._build_auto_input(auto_inputs)
        print("The inputs of model are {}".format(inputs))
        self._check_input(inputs)
        inputs = self._convert_data_format(inputs)
        compare_results = []
        if self.fix_random_seed is not None:
            self._fix_random(self.fix_random_seed)
        if self.auto_conv_ckpt:
            self._conv_compare_ckpt()
            self._load_conve_ckpt()
        elif self.pt_path and self.ms_path:
            self.pt_net.load_state_dict(torch.load(self.pt_path))
            ms.load_param_into_net(self.ms_net, 
                                   ms.load_checkpoint(self.ms_path))
        else:
            logger.user_warning("For 'ts.migrator.NetDifferenceFinder', \
                please ensure that the network has been initialized when \
                    auto_conv_ckpt is set to False and ms_path/pt_path is empty.")
        for idx, input in enumerate(inputs):
            input_data = self._get_input_data(input)
            result_ms, result_pt = self._infer_net(
                input_data, self.ms_net, self.pt_net, idx)
            self._check_output(result_ms, result_pt)
            compare_results.extend(
                self._compare_results(
                    result_ms, 
                    result_pt, 
                    idx, 
                    atol, 
                    rtol, 
                    equal_nan))
        self._save_results(compare_results)
        
        print_diff_result(compare_results, title=TITLE, field_names=FIELD_NAMES)

    def _get_input_data(self, input):
        input_data = []
        if isinstance(input, tuple): # for multiple inputs
            for data in input:
                if isinstance(data, str):
                    input_data.append(np.load(data))
                elif isinstance(data, np.ndarray):
                    input_data.append(data)
                elif isinstance(data, torch.Tensor):
                    input_data.append(data.numpy())
                elif isinstance(data, ms.Tensor):
                    input_data.append(data.asnumpy())
                else:
                    raise TypeError('Unknow input data type {}'.format(type(data)))
        return input_data

    def _fix_random(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        # for torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        # for mindspore
        ms.set_seed(seed)

    def _save_ckpt(self):
        torch.save(self.pt_net.state_dict(), self.pt_org_pth)
        ms.save_checkpoint(self.ms_net, self.ms_org_ckpt)

    def _conv_compare_ckpt(self):
        self._save_ckpt()

        weight_map = weight_migrator.get_weight_map(pt_model=self.pt_net, print_map=True)
        weight_migrator.convert_weight(weight_map=weight_map, pt_file_path=self.pt_org_pth,
                                       ms_file_save_path=self.conv_ckpt_name, print_level=0)
        weight_migrator.compare_ms_ckpt(orig_file_path=self.conv_ckpt_name,
                                        target_file_path=self.ms_org_ckpt,
                                        compare_value=True,
                                        weight_map=weight_map)

    def _load_conve_ckpt(self):
        param_dict = ms.load_checkpoint(self.conv_ckpt_name)
        ms.load_param_into_net(self.ms_net, param_dict)

    def _infer_net(self, input_data, ms_net, pt_net, idx):
        if self.print_result:
            print("\n=================================Start inference net=================================")
        start_pt = time.time()
        result_pt = self._run_pt_net(pt_net, input_data)
        end_pt = time.time()
        print(f"In test case {idx}, the PyTorch net inference completed cost %.5f seconds." % (
                end_pt - start_pt))
        start_ms = time.time()
        result_ms = self._run_ms_net(ms_net, input_data)
        end_ms = time.time()
        print(f"In test case {idx}, the MindSpore net inference completed cost %.5f seconds." % (
            end_ms - start_ms))
        if isinstance(result_ms, (tuple, list)):
            result_ms = {f"result_{idx}": result for idx,
                         result in enumerate(result_ms)}
        elif isinstance(result_ms, ms.Tensor):
            result_ms = {"result": result_ms}
        if isinstance(result_pt, (tuple, list)):
            result_pt = {f"result_{idx}": result for idx,
                         result in enumerate(result_pt)}
        elif torch.is_tensor(result_pt):
            result_pt = {"result": result_pt}
        return result_ms, result_pt

    def _run_pt_net(self, pt_net, input_data_list):
        data_list = []
        for data in input_data_list:
            data_list.append(torch.tensor(data))
        pt_results = pt_net(*data_list)
        return pt_results

    def _run_ms_net(self, ms_net, input_data_list):
        data_list = []
        for data in input_data_list:
            data_list.append(ms.Tensor(data))
        ms_results = ms_net(*data_list)
        return ms_results

    def _check_output(self, result_ms, result_pt):
        ms_result_num = len(result_ms)
        if self.print_result:
            print("The MindSpore net inference have %s result." % ms_result_num)
        pt_result_num = len(result_pt)
        if self.print_result:
            print("The PyTorch net inference have %s result." % pt_result_num)
        if ms_result_num != pt_result_num:
            raise ValueError("output results are in different counts!")

    def _compare_results(self, result_ms, result_pt, idx, atol, rtol, equal_nan):
        index = 0
        compare_result = []
        for (k_pt, k_ms) in zip(result_pt, result_ms):
            try:
                result_pt_ = result_pt[k_pt].detach().numpy()
                result_ms_ = result_ms[k_ms].asnumpy()
                self._check_out_data_shape(index, result_ms_, result_pt_)
                self._save_out_data(index, k_ms, k_pt, result_ms_, result_pt_)
                result_pt_ = result_pt_.reshape(1, -1)
                result_ms_ = result_ms_.reshape(1, -1)
                OUTPUT_TYPE_ERROR = "The type of output is not as expected, \
                    the type of output should be numpy.ndarray or torch.Tensor or mindspore.Tensor." 
                if not isinstance(result_ms_, np.ndarray):
                    raise TypeError(OUTPUT_TYPE_ERROR)
                if not isinstance(result_pt_, np.ndarray):
                    raise TypeError(OUTPUT_TYPE_ERROR)
                result, rel_ratio, cosine_sim, diff_detail = cal_algorithm(result_ms_, result_pt_, rtol,
                                                                           atol, equal_nan)
                org_name, targ_name = f"test{idx}-{k_ms}", f"test{idx}-{k_pt}"
                compare_result.append((org_name, targ_name,result, rel_ratio, cosine_sim, diff_detail))
            except Exception as e:
                logger.error(e)
                logger.user_warning("The returned result cannot be compared normally, "
                                    "the result index is " + str(index))
            index += 1
        return compare_result

    def _check_out_data_shape(self, index, result_ms, result_pt):
        if self.print_result:
            print("\n=========================Start Check Out Data %s ===========================" % index)
        pt_out_shape = result_pt.shape
        ms_out_shape = result_pt.shape
        if not pt_out_shape == ms_out_shape:
            raise ValueError("output results are in different shapes!")
        if self.print_result:
            print("shape of result_pt: %s" % str(pt_out_shape))
            print("shape of result_ms: %s" % str(ms_out_shape))
            print("-result_pt-: \n", result_pt)
            print("-result_ms-: \n", result_ms)

    def _save_out_data(self, index, k_ms, k_pt, result_ms, result_pt):
        if self.print_result:
            print(
                "\n=========================Start Save Out Data %s ===========================" % index)
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

    def _save_results(self, compare_results):
        if self.print_result:
            logger.info(
                "=================================Start save result=================================")
        result_path = os.path.join(self.out_path, "compare_result.csv")
        with open(result_path, "w", encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(FIELD_NAMES)
            writer.writerows(compare_results)
            logger.info(
                "The comparison result have been written to %s" % result_path)
