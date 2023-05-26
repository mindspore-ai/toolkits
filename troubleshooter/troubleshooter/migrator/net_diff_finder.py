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

    def __init__(self, pt_net, ms_net, inputs=None, auto_input=None, **kwargs):
        self.ms_net = ms_net
        self.pt_net = pt_net
        self.inputs = inputs
        self.auto_input = auto_input
        self.print_result = kwargs.get('print_result', True)
        self.auto_conv_ckpt = kwargs.get('auto_conv_ckpt', True)
        self.fix_random_sed = kwargs.get('fix_random_sed', 16)
        self.out_path = validate_and_normalize_path(kwargs.get('out_path', './'))
        self.pt_org_pth = os.path.join(self.out_path, 'net_diff_finder_pt_org_pth.pth')
        self.conv_ckpt_name = os.path.join(self.out_path, 'net_diff_finder_conv_ckpt.ckpt')
        self.ms_org_ckpt = os.path.join(self.out_path, 'net_diff_finder_ms_org_ckpt.ckpt')
        self.rtol = None
        self.atol = None
        self.equal_nan = None

    def build_auto_input(self):
        self.inputs = list()
        if self.auto_input:
            if isinstance(self.auto_input, tuple):
                for item in self.auto_input:
                    assert isinstance(item[0], tuple), "The shape of input data must be tuple"
                    self.inputs.append(np.random.randn(*item[0]).astype(item[1]))
                self.inputs = tuple(self.inputs)
            elif isinstance(self.auto_input, dict):
                assert 'input' in self.auto_input.keys(), "The keys of auto_input must contain 'input'"
                assert 'num' in self.auto_input.keys(), "The keys of auto_input must contain 'num'"
                tmp = []
                for item in self.auto_input['input']:
                    assert isinstance(item[0], tuple), "The shape of input data must be tuple"
                    tmp.append(np.random.randn(*item[0]).astype(item[1]))
                tmp = tuple(tmp)
                self.inputs = [tmp] * self.auto_input['num']
            else:
                logger.user_error(
                    'Unknow auto_input type {}'.format(type(self.auto_input)))
                exit(1)
    
    def convert_data_format(self):
        inputs = []
        if isinstance(self.inputs, tuple):
            inputs.append(self.inputs)
        elif isinstance(self.inputs, list):
            inputs = self.inputs
        self.inputs = inputs
        
    def compare(self, **kwargs):
        if self.auto_input:
            self.build_auto_input()
        self.convert_data_format()
        compare_results = []
        self.rtol = kwargs.get('rtol', 1e-04)
        self.atol = kwargs.get('atol', 1e-08)
        self.equal_nan = kwargs.get('equal_nan', False)
        if self.fix_random_sed is not None:
            self._fix_random(self.fix_random_sed)
        if self.auto_conv_ckpt:
            self._conv_compare_ckpt();
            self._load_conve_ckpt();
        for idx, input in enumerate(self.inputs):
            input_data = self.get_input_data(input)
            result_ms, result_pt = self.infer_net(
                input_data, self.ms_net, self.pt_net, idx)
            self.check_output(result_ms, result_pt)
            compare_results.extend(
                self.compare_results(result_ms, result_pt, idx))
        self.save_results(compare_results)
        
        print_diff_result(compare_results, title=TITLE, field_names=FIELD_NAMES)

    def get_input_data(self, input):
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
                    logger.user_error(
                        'Unknow input data type {}'.format(type(data)))
                    exit(1)
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

    def infer_net(self, input_data, ms_net, pt_net, idx):
        if self.print_result:
            print("\n=================================Start inference net=================================")
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
            try:
                result_pt_ = result_pt[k_pt].detach().numpy()
                result_ms_ = result_ms[k_ms].asnumpy()
                self.check_out_data_shape(index, result_ms_, result_pt_)
                self.save_out_data(index, k_ms, k_pt, result_ms_, result_pt_)
                result_pt_ = result_pt_.reshape(1, -1)
                result_ms_ = result_ms_.reshape(1, -1)
                OUTPUT_TYPE_ERROR = "The type of output is not as expected, \
                    the type of output should be numpy.ndarray or torch.Tensor or mindspore.Tensor." 
                assert isinstance(result_ms_, np.ndarray), OUTPUT_TYPE_ERROR
                assert isinstance(result_pt_, np.ndarray), OUTPUT_TYPE_ERROR
                result, rel_ratio, cosine_sim, diff_detail = cal_algorithm(result_ms_, result_pt_, self.rtol,
                                                                           self.atol, self.equal_nan)
                org_name, targ_name = f"test{idx}-{k_ms}", f"test{idx}-{k_pt}"
                compare_result.append((org_name, targ_name,result, rel_ratio, cosine_sim, diff_detail))
            except Exception as e:
                logger.error(e)
                logger.user_warning("The returned result cannot be compared normally, "
                                    "the result index is " + str(index))
            index += 1
        return compare_result

    def check_out_data_shape(self, index, result_ms, result_pt):
        if self.print_result:
            print("\n=========================Start Check Out Data %s ===========================" % index)
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

    def save_results(self, compare_results):
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
