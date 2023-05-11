import os
import csv
import time
import torch
import numpy as np
from troubleshooter import log as logger
from troubleshooter.migrator.diff_handler import cal_similarity
from troubleshooter.common.util import save_numpy_data
from troubleshooter.common.format_msg import print_net_infer_diff_result

try:
    import mindspore as ms
except ModuleNotFoundError as e:
    e_msg = e.msg
    no_module_msg = "No module named 'mindspore'"
    if e_msg == no_module_msg:
        FRAMEWORK_TYPE = 'pt'
    else:
        raise e

MS_OUTPUT_PATH = "data/output/MindSpore"
PT_OUTPUT_PATH = "data/output/PyTorch"
RESULT_COLUMNS = ["Pytorch data", "MindSpore data",
                  "Results of comparison", "cosine similarity", "(mean, max, min)"]

class NetDifferenceFinder:

    def __init__(self, ms_net, pt_net, inputs, out_path, print_result):
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
        if isinstance(result_ms, (tuple, list, ms.Tensor)):
            result_ms = {f"result_{idx}": result for idx,
                         result in enumerate(result_ms)}
        if isinstance(result_pt, (tuple, list)) or torch.is_tensor(result_pt):
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
            try:
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
            except:
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
