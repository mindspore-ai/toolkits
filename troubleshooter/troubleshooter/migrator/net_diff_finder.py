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
import os
import random
import time
import shutil
import tempfile

import torch
import mindspore as ms
import numpy as np
from troubleshooter import log as logger
from troubleshooter.common.format_msg import print_diff_result, print_separator_line
from troubleshooter.common.util import save_numpy_data, validate_and_normalize_path, clear_tmp_file, type_check
from troubleshooter.migrator import weight_migrator
from troubleshooter.migrator.diff_handler import cal_algorithm


__all__ = [
    "NetDifferenceFinder",
]


class NetDifferenceFinder:
    init_kwargs = {"pt_params_path", "ms_params_path", "auto_conv_ckpt", "fix_seed", "out_path", "pt_org_pth_name",
                   "conv_ckpt_name", "ms_conv_ckpt_name"}
    TITLE = 'The comparison results of Net'
    FIELD_NAMES = ["PyTorch out", "MindSpore out",
                "results of comparison", "match ratio", "cosine similarity", "(mean, max)"]
    def __init__(self,
                 pt_net,
                 ms_net,
                 print_level=1,
                 **kwargs):
        self.ms_net = ms_net
        self.pt_net = pt_net
        self.print_level = print_level
        self.out_path = tempfile.mkdtemp(prefix="tmp_net_diff_finder_")
        self._handle_kwargs(**kwargs)
        if self.fix_seed:
            self.fix_random(self.fix_seed)

    def __del__(self):
        shutil.rmtree(self.out_path)

    def _handle_kwargs(self, **kwargs):

        self.pt_params_path = kwargs.get('pt_params_path', None)
        self.ms_params_path = kwargs.get('ms_params_path', None)
        self.auto_conv_ckpt = kwargs.get('auto_conv_ckpt', 1)
        self.compare_params = kwargs.get('compare_params', True)

        self.fix_seed = kwargs.get('fix_seed', 16)

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        self.pt_org_pth_name = os.path.join(
            self.out_path,
            kwargs.get('pt_org_pth_name', 'net_diff_finder_pt_org_pth.pth'))
        self.conv_ckpt_name = os.path.join(
            self.out_path,
            kwargs.get('conv_ckpt_name', 'net_diff_finder_conv_ckpt.ckpt'))
        self.ms_conv_ckpt_name = os.path.join(
            self.out_path,
            kwargs.get('ms_conv_ckpt_name', 'net_diff_finder_ms_conv_ckpt.ckpt'))

    def _load_and_convert(self, pt_params_path, ms_params_path, auto_conv_ckpt):
        if pt_params_path and ms_params_path:
            validate_and_normalize_path(pt_params_path)
            validate_and_normalize_path(ms_params_path)
            self.pt_net.load_state_dict(torch.load(pt_params_path))
            ms.load_param_into_net(self.ms_net,
                                   ms.load_checkpoint(ms_params_path))
        elif pt_params_path:
            validate_and_normalize_path(pt_params_path)
            self.pt_net.load_state_dict(torch.load(pt_params_path))
            if auto_conv_ckpt:
                self._convert_ckpt()
                self._load_and_save()
            else:
                logger.user_warning("For 'ts.migrator.NetDifferenceFinder', \
                    please ensure that the network has been initialized when \
                    'auto_conv_ckpt' is set to 'False' and 'ms_params_path' is empty.")
        elif ms_params_path:
            validate_and_normalize_path(ms_params_path)
            ms.load_param_into_net(self.ms_net,
                                   ms.load_checkpoint(ms_params_path))
            logger.error("For 'ts.migrator.NetDifferenceFinder', \
                please ensure that the network has been initialized when \
                'auto_conv_ckpt' is set to 'False' 'pt_params_path' is empty.")
        elif auto_conv_ckpt:
            self._convert_ckpt()
            self._load_and_save()
        else:
            logger.error("For 'ts.migrator.NetDifferenceFinder', \
                please ensure that the network has been initialized when \
                'auto_conv_ckpt' is set to 'False' and 'ms_params_path' / 'pt_params_path' is empty.")

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

    def _load_and_save(self):
        param_dict = ms.load_checkpoint(self.conv_ckpt_name)
        ms.load_param_into_net(self.ms_net, param_dict)
        ms.save_checkpoint(self.ms_net, self.ms_conv_ckpt_name)

    def _check_auto_input(self, auto_inputs):
        if auto_inputs is None:
            return None
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
        inputs = []
        if auto_inputs is None:
            return None

        if isinstance(auto_inputs, tuple):
            for item in auto_inputs:
                inputs.append(np.random.randn(*item[0]).astype(item[1]))
            inputs = tuple(inputs)
        elif isinstance(auto_inputs, dict):
            type_check(auto_inputs["num"], "auto_inputs['num']", int)
            for _ in range(auto_inputs["num"]):
                inputs.append(self._build_auto_input(auto_inputs["input"]))
        else:
            raise ValueError("For ts.migrator.NetDifferenceFinder, the arg 'auto_inputs' value error.")
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
                rtol=1e-4,
                atol=1e-4,
                equal_nan=False):
        self._check_auto_input(auto_inputs)
        self._load_and_convert(self.pt_params_path, self.ms_params_path, self.auto_conv_ckpt)
        self._compare_ckpt()
        if auto_inputs:
            inputs = self._build_auto_input(auto_inputs)
        self._check_input(inputs)
        inputs = self._convert_data_format(inputs)
        compare_results = self._inference(inputs, rtol, atol, equal_nan)
        print_diff_result(compare_results, title=self.TITLE, field_names=self.FIELD_NAMES)

    def _inference(self, inputs, rtol, atol, equal_nan):
        compare_results = []
        if self.print_level:
            print_separator_line("Start Performing Network Inference", length=141)
        for idx, input in enumerate(inputs):
            input_data = self._get_input_data(input)
            if self.print_level:
                print_separator_line(f"Start case {idx} inference", length=141, character='-')
            result_ms, result_pt = self._infer_net(input_data, self.ms_net, self.pt_net, idx)
            if len(result_ms) != len(result_pt):
                raise ValueError("Output results are in different counts! "
                                 f"MindSpore output num: {len(result_ms)}, PyTorch output shape: {len(result_pt)}.")
            compare_results.extend(self._compare_results(result_ms, result_pt, idx, atol, rtol, equal_nan))
        return compare_results

    def _get_input_data(self, input):
        input_data = []
        if isinstance(input, tuple):
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

    def fix_random(self, seed=16):
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
        self.pt_net.eval()
        self.ms_net.set_train(False)

    def _convert_ckpt(self):
        torch.save(self.pt_net.state_dict(), self.pt_org_pth_name)
        if self.auto_conv_ckpt == 0:
            return
        if self.auto_conv_ckpt == 1:
            self.weight_map = weight_migrator.get_weight_map(pt_model=self.pt_net, print_map=False)
            if self.print_level:
                print_separator_line("Start Converting PyTorch Weights to MindSpore", length=141)
            weight_migrator.convert_weight(weight_map=self.weight_map,
                                           pt_file_path=self.pt_org_pth_name,
                                           ms_file_save_path=self.conv_ckpt_name,
                                           print_level=self.print_level,
                                           print_save_path=False)
        elif self.auto_conv_ckpt == 2:
            self._msadapter_pth2ckpt(self.pt_org_pth_name, self.conv_ckpt_name)
        else:
            raise ValueError("For 'NetDifferenceFinder',  the parameter 'auto_conv_ckpt' "
                             f"currently only supports 0, 1, and 2, but got {self.auto_conv_ckpt}.")

    def _compare_ckpt(self):
        if self.compare_params is False:
            return
        shape_field_names = ["Parameter name of torch", "Parameter name of MindSpore", "Whether shape are equal",
                             "Parameter shape of torch", "Parameter shape of MindSpore"]
        value_field_names = ["Parameter name of torch", "Parameter name of MindSpore", "results of comparison",
                             "match ratio", "cosine similarity", "(mean, max)"]
        if self.auto_conv_ckpt == 1:
            print_separator_line("Start comparing PyTorch and MindSpore parameters", length=141)
            weight_migrator.compare_ms_ckpt(orig_file_path=self.conv_ckpt_name,
                                            target_file_path=self.ms_conv_ckpt_name,
                                            compare_value=True, weight_map=self.weight_map,
                                            shape_field_names=shape_field_names,
                                            value_field_names=value_field_names)
        elif self.auto_conv_ckpt == 2:
            print_separator_line("Start comparing PyTorch and MSAdapter parameters", length=141)
            weight_migrator.compare_ms_ckpt(orig_file_path=self.conv_ckpt_name,
                                            target_file_path=self.ms_conv_ckpt_name,
                                            compare_value=True,
                                            shape_field_names=shape_field_names,
                                            value_field_names=value_field_names)
        else:
            raise ValueError("For 'NetDifferenceFinder', when the argument 'auto_conv_ckpt' is 0, "
                             "the argument 'compare_params' is not supported.")

    def _load_conve_ckpt(self):
        param_dict = ms.load_checkpoint(self.conv_ckpt_name)
        ms.load_param_into_net(self.ms_net, param_dict)
        clear_tmp_file(self.conv_ckpt_name)

    def _infer_net(self, input_data, ms_net, pt_net, idx):
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
            result_ms = {f"out_{idx}": result for idx, result in enumerate(result_ms)}
        elif isinstance(result_ms, ms.Tensor):
            result_ms = {"out": result_ms}
        if isinstance(result_pt, (tuple, list)):
            result_pt = {f"out_{idx}": result for idx, result in enumerate(result_pt)}
        elif torch.is_tensor(result_pt):
            result_pt = {"out": result_pt}
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


    def _compare_results(self, result_ms, result_pt, idx, atol, rtol, equal_nan):
        compare_result = []
        for index, (k_pt, k_ms) in enumerate(zip(result_pt, result_ms)):
            try:
                result_pt_ = result_pt[k_pt].detach().numpy()
                result_ms_ = result_ms[k_ms].asnumpy()
                if result_pt_.shape != result_ms_.shape:
                    raise ValueError(f"Output results for {index} have different shapes! "
                                     f"MindSpore output shape: {result_ms_.shape}, PyTorch output shape: {result_pt_.shape}.")
                result, rel_ratio, cosine_sim, diff_detail = cal_algorithm(result_ms_, result_pt_, rtol, atol, equal_nan)
                pt_name, ms_name = f"step_{idx}_{k_pt}", f"step_{idx}_{k_ms}"
                compare_result.append((pt_name, ms_name, result, rel_ratio, cosine_sim, diff_detail))
            except Exception as e:
                logger.error(e)
                logger.user_warning("The returned result cannot be compared normally, "
                                    "the result index is " + str(index))
        return compare_result

    def _save_out_data(self, k_ms, k_pt, result_ms, result_pt):
        result_file = "%s.npy" % k_ms
        ms_out_path = os.path.join(self.out_path, self.MS_OUTPUT_PATH, result_file)
        save_numpy_data(ms_out_path, result_ms)
        if self.print_level:
            print("Saved MindSpore output data at: %s" % ms_out_path)

        result_file = "%s.npy" % k_pt
        pt_out_path = os.path.join(self.out_path, self.PT_OUTPUT_PATH, result_file)
        save_numpy_data(pt_out_path, result_pt)
        if self.print_level:
            print("Saved PyTorch output data at: %s" % pt_out_path)

    def _save_results(self, compare_results):
        result_path = os.path.join(self.out_path, "compare_result.csv")
        with open(result_path, "w", encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.FIELD_NAMES)
            writer.writerows(compare_results)
            logger.info(
                "The comparison result have been written to %s" % result_path)

    def _msadapter_pth2ckpt(self, pt_path, ms_path):
        torch_dict = torch.load(pt_path, map_location='cpu')
        ms_params = []
        for name, value in torch_dict.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    param_dict = {}
                    param_dict['name'] = k
                    if isinstance(v, torch.Tensor):
                        param_dict['data'] = ms.Tensor(v.detach().cpu().numpy())
                    else:
                        param_dict['data'] = ms.Tensor(v)
                    ms_params.append(param_dict)
                continue
            else:
                param_dict = {}
                param_dict['name'] = name
                if isinstance(value, torch.Tensor):
                    param_dict['data'] = ms.Tensor(value.detach().cpu().numpy())
                else:
                    param_dict['data'] = ms.Tensor(value)
                ms_params.append(param_dict)

        ms.save_checkpoint(ms_params, ms_path)
