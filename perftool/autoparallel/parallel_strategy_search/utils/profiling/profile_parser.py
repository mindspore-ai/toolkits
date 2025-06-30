# Copyright 2024 Huawei Technologies Co., Ltd
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

import json
import os
import statistics
import csv
from typing import List
from utils.logger import logger
from utils.ppc_input import PipelineInputConfig
from pathlib import Path

encoding = 'utf-8'

def find_file_by_name(directory, filename):
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None

def median_mean(durs):
    if len(durs) == 0:
        logger.info(f"get durs no values")
        return 0
    median_durs = durs[len(durs)//4 : len(durs) - len(durs)//4]
    return round(statistics.mean(median_durs)/1000, 3)

class ProfileParser:

    '''
    The recommend config for different pp is：
    pp2: [6,5] recomp [6,0], reduce laysers when OOM, layers at least [5,4]
    if still OOM, do another profile wrt [3,1] no recomp w/ dense_replace = 1
    pp4: [3,3,3,2] recomp [3,3,0,0]
    pp8: [3,1,1,1,1,2,2,2] recomp [3,1,1,1,1,2,0,2] (or try pp4 with layers [3,2,2,2] w/ recomp [3,2,0,2])
    a typical input is a string and a tuple of numbers. 2-tuple for pp2 and 4-tuple for higher pp,
    which consists the rank for each stages.
    '''

    profile_data = None
    config = ''
    recomp_bws, moe_fws1, moe_fws2, moe_bws, dense_fws, dense_bws, totals = [], [], [], [], [], [], []
    moe_fw1, moe_fw2, moe_bw, recomp_bw, dense_fw, dense_bw, head_ratio, num_last_stage_layers, mtp = 0,0,0,0,0,0,0,0,0
    att_tid, grad_att_tid = 2, 2

    def __init__(self, mbn, layer_num):
        self.mbn = mbn
        self.layer_num = layer_num
        pass

    def parse_batch_profile_result(self, profile_configs_results, para):
        # step1. 解析profile文件
        # ep填1, dmratrion:0.33, bfratio=dense_bw/dense_fw, re_grow_ration = 1,hratio=0.68
        # [dp, tp, pp, dmratio, bfratio, re_grow_ration, hratio, moe_fw]
        profile_result = []
        candidate_configs: List[PipelineInputConfig] = []
        for result in profile_configs_results:
            profile_dir = result[-1]
            pp = result[2]
            config = result[:3]
            config += [0.33, 0, 1, 0.68, 0]
            self.config_anal_llama(profile_dir, pp, config)
            self.refresh()
            profile_result.append(config)

        # step2. 生成csv文件
        self.write_result_to_csv(profile_result, para, True)
        return candidate_configs

    # 把profile结果写入csv
    def write_result_to_csv(self, profile_result, para, is_llama=False):
        if is_llama:
            headers = ['dp', 'tp', 'pp', 'dmratio', 'bfratio', 're_grow_ration', 'hratio', 'moe_fw']
        else:
            headers = ['dp', 'tp', 'pp', 'ep', 'dmratio', 'bfratio', 're_grow_ration', 'hratio', 'moe_fw']
        # 写入 CSV 文件
        try:
            csv_path = os.path.join(os.path.abspath(para.OUTPUT_PATH), 'profile_parser_result.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # 写入表头
                writer.writerow(headers)
                # 写入数据
                writer.writerows(profile_result)
            logger.info(f"CSV file {csv_path} generate succ.")
            para.args.parser_result = csv_path
        except Exception as e:
            print(f"write CSV file fail: {e}")

    def load_profile(self, file, rk):
        global encoding
        path = os.path.abspath('.\\profile_info') + '\\' + str(file)
        path = find_file_by_name(path + '\\rank_' + str(rk), 'trace_view.json')
        with open(path, 'r', encoding=encoding) as f:
            self.profile_data = json.load(f)

    def load_profile_by_dir(self, rank_dir):
        global encoding
        path = find_file_by_name(rank_dir, 'trace_view.json')
        with open(path, 'r', encoding=encoding) as f:
            self.profile_data = json.load(f)

    def load_profile_no_recompute(self, file, rk):
        global encoding
        path = os.path.abspath('.\\profile_info') + '\\' + str(file)
        path = find_file_by_name(path + '\\rank_' + str(rk) + '_norecomp' , 'trace_view.json')
        with open(path, 'r', encoding=encoding) as f:
            self.profile_data = json.load(f)

    def refresh(self):
        self.profile_data = None
        self.recomp_bws, self.moe_fws1, self.moe_fws2, self.moe_bws, self.dense_bws, self.totals = [[] for _ in range(6)]
        self.moe_fw1, self.moe_fw2, self.moe_bw, self.recomp_bw, self.dense_fw, self.dense_bw, self.head_ratio, self.num_last_stage_layers, self.mtp = 0, 0, 0, 0, 0, 0, 0, 0, 0

    def release(self):
        self.profile_data = None

    def set_tid(self, a, g):
        self.att_tid, self.grad_att_tid = a, g

    def extract_atts(self):
        atts, grad_atts = [], []
        for line in self.profile_data:
            name = line['name']
            if line['tid'] == self.att_tid and name.endswith('FlashAttentionScore_FlashAttentionScore'):
                atts.append(float(line['ts']))
            if line['tid'] == self.grad_att_tid and name.endswith('FlashAttentionScoreGrad_FlashAttentionScoreGrad'):
                grad_atts.append(float(line['ts']))
        logger.info(f'num of atts is {len(atts)}, num of grad_atts is {len(grad_atts)}')
        return atts, grad_atts

    def stage_anal_llama(self, pp, stage, isRecompute=False): #assuming we profiled 2 steps w/ micro = 32
        atts, grad_atts = self.extract_atts()
        step = 2
        mbn = self.mbn
        net_type = 0 #0表示llama, 1表示deepseek
        div = step * mbn
        flag = 2 if isRecompute else 1
        layers, xlayers = len(grad_atts) // div,  flag * len(grad_atts) // div
        warm_up = (pp-1-stage) * layers
        atts = atts[warm_up : mbn*xlayers - warm_up] + atts[mbn*xlayers + warm_up : mbn*xlayers + mbn*xlayers-warm_up]
        grad_atts = grad_atts[warm_up : mbn*layers - warm_up] + grad_atts[mbn*layers + warm_up : mbn*layers + mbn*layers-warm_up]
        att_chunks = [atts[xlayers*i:xlayers*(i+1)] for i in range(len(atts)//xlayers-1)]
        grad_chunks = [grad_atts[layers*i:layers*(i+1)] for i in range(len(grad_atts)//layers-1)]
        if pp == 2 or net_type == 0:
            if stage == 0:
                for chunk in att_chunks:
                    for i in range(layers-1):
                        if net_type == 1:
                            if i <= 0: self.dense_fws.append(chunk[i+1] - chunk[i])
                            else: self.moe_fws1.append(chunk[i+1] - chunk[i])
                        else:
                            self.dense_fws.append(chunk[i+1] - chunk[i])
                for chunk in grad_chunks:
                    for i in range(layers-2, layers-1): self.dense_bws.append(chunk[i+1] - chunk[i])
                    if net_type == 1:
                        for i in range(layers-3): self.recomp_bws.append(chunk[i+1] - chunk[i])
                for i in range(len(att_chunks)-1): self.totals.append(att_chunks[i+1][0] - att_chunks[i][0])
            if stage == pp - 1:
                self.num_last_stage_layers = len(grad_chunks[0])
                for chunk in att_chunks:
                    for i in range(layers-2): self.moe_fws2.append(chunk[i+1] - chunk[i])
                for chunk in grad_chunks:
                    for i in range(2, layers-1): self.moe_bws.append(chunk[i+1] - chunk[i])
                for i in range(len(att_chunks) - 1): self.totals.append(att_chunks[i + 1][0] - att_chunks[i][0])

        else:
            if stage == 0:
                for chunk in att_chunks:
                    for i in range(layers-1):
                        if i <= 2: self.dense_fws.append(chunk[i+1] - chunk[i])
                for chunk in grad_chunks:
                    for i in range(layers-3, layers-1): self.dense_bws.append(chunk[i+1] - chunk[i])
            elif stage == pp - 3:
                for chunk in att_chunks:
                    for i in range(layers-1):
                        self.moe_fws1.append(chunk[i+1] - chunk[i])
                for chunk in grad_chunks:
                    for i in range(layers-3, layers-1): self.recomp_bws.append(chunk[i+1] - chunk[i])
            elif stage == pp - 2:
                for chunk in att_chunks:
                    for i in range(layers-1): self.moe_fws2.append(chunk[i+1] - chunk[i])
                for chunk in grad_chunks:
                    for i in range(layers-3, layers-1): self.moe_bws.append(chunk[i+1] - chunk[i])
            else:
                for i in range(len(att_chunks)-1): self.totals.append(att_chunks[i+1][0] - att_chunks[i][0])
                self.num_last_stage_layers = len(grad_chunks[0])

    #todo: 待修改
    def stage_anal(self, pp, stage, isRecompute): #assuming we profiled 2 steps w/ micro = 32
        atts, grad_atts = self.extract_atts()
        layers, xlayers = len(grad_atts) // 64,  2 * len(grad_atts) // 64
        warm_up = (pp-1-stage) * layers
        atts = atts[warm_up : 32*xlayers - warm_up] + atts[32*xlayers + warm_up : 32*xlayers + 32*xlayers-warm_up]
        grad_atts = grad_atts[warm_up : 32*layers - warm_up] + grad_atts[32*layers + warm_up : 32*layers + 32*layers-warm_up]
        att_chunks = [atts[xlayers*i:xlayers*(i+1)] for i in range(len(atts)//xlayers-1)]
        grad_chunks = [grad_atts[layers*i:layers*(i+1)] for i in range(len(grad_atts)//layers-1)]
        if pp == 2:
            if stage == 0:
                for chunk in att_chunks:
                    for i in range(layers-1):
                        if i <= 2: self.dense_fws.append(chunk[i+1] - chunk[i])
                        else: self.moe_fws1.append(chunk[i+1] - chunk[i])
                for chunk in grad_chunks:
                    for i in range(layers-3, layers-1): self.dense_bws.append(chunk[i+1] - chunk[i])
                    for i in range(layers-4): self.recomp_bws.append(chunk[i+1] - chunk[i])
                for i in range(len(att_chunks)-1): self.totals.append(att_chunks[i+1][0] - att_chunks[i][0])
            if stage == pp - 1:
                for chunk in att_chunks:
                    for i in range(layers-2): self.moe_fws2.append(chunk[i+1] - chunk[i])
                for chunk in grad_chunks:
                    for i in range(2, layers-1): self.moe_bws.append(chunk[i+1] - chunk[i])
                for i in range(len(att_chunks) - 1): self.totals.append(att_chunks[i + 1][0] - att_chunks[i][0])
                self.num_last_stage_layers = layers
        else:
            if stage == 0:
                for chunk in att_chunks:
                    for i in range(layers-1):
                        if i <= 2: self.dense_fws.append(chunk[i+1] - chunk[i])
                for chunk in grad_chunks:
                    for i in range(layers-3, layers-1): self.dense_bws.append(chunk[i+1] - chunk[i])
            elif stage == pp - 3:
                for chunk in att_chunks:
                    for i in range(layers-1):
                        self.moe_fws1.append(chunk[i+1] - chunk[i])
                for chunk in grad_chunks:
                    for i in range(layers-3, layers-1): self.moe_bws.append(chunk[i+1] - chunk[i])
            elif stage == pp - 2:
                for chunk in att_chunks:
                    for i in range(layers-1): self.moe_fws2.append(chunk[i+1] - chunk[i])
                for chunk in grad_chunks:
                    for i in range(layers-3, layers-1): self.moe_bws.append(chunk[i+1] - chunk[i])
            else:
                for i in range(len(att_chunks)-1): self.totals.append(att_chunks[i+1][0] - att_chunks[i][0])
                self.num_last_stage_layers = layers

    def config_anal(self, config, ranks):
        self.config = config
        pp = len(ranks)
        for i in range(pp):
            self.load_profile(config, ranks[i])
            self.stage_anal(pp, i)
            self.release()

    def config_anal_llama(self, profile_dir, pp, profile_result):
        self.config = profile_dir
        self.process_folders(profile_dir, pp, profile_result)

    def process_folders(self, profile_result_dir, pp, profile_result):
        """
        遍历根目录下的所有文件夹，并使用 parser 函数解析每个文件夹中的内容

        参数:
        profile_result_dir (str): 根目录路径
        """
        root_path = Path(profile_result_dir)
        if not root_path.is_dir():
            logger.info(f"profile_result_dir:{profile_result_dir} not exist")
            return

        # 获取所有子文件夹--每个rank的文件夹
        folders = [f for f in root_path.iterdir() if f.is_dir()]
        i = 0
        for rank_folder in folders:
            logger.info(f"parsing: {rank_folder}")
            self.load_profile_by_dir(rank_folder)
            self.stage_anal_llama(pp, i)
            i += 1
        self.refined_data()
        self.fill_result(profile_result)

    def data_display(self):
        logger.info(f'moe_fws1 are {self.moe_fws1}')
        logger.info(f'moe_fws2 are {self.moe_fws2}')
        logger.info(f'moe_bws is {self.moe_bws}')
        logger.info(f'recomp_bws is {self.recomp_bws}')
        logger.info(f'dense_fw is {self.dense_fw}')
        logger.info(f'dense_bw is {self.dense_bws}')
        logger.info(f'totals is {self.totals}')

    def refined_data(self):
        if len(self.dense_fws) == 0 or len(self.dense_bws) == 0:
            logger.info(f"dense_fws len {len(self.dense_fws)} "
                        f"or dense_bws {len(self.dense_fws)} is empty")
            return
        self.moe_fw1 = median_mean(self.moe_fws1)
        self.moe_fw2 = median_mean(self.moe_fws2)
        self.moe_bw = median_mean(self.moe_bws)
        self.recomp_bw = median_mean(self.recomp_bws)
        self.dense_fw = median_mean(self.dense_fws)
        self.dense_bw = median_mean(self.dense_bws)
        total = median_mean(self.totals)
        llama_cost = self.mbn * (self.dense_fw + self.dense_bw) * self.layer_num
        logger.info(f'config is {self.config}, dense forward is {self.dense_fw}, dense backward is {self.dense_bw}, llama cost is {llama_cost}')

    def fill_result(self, profile_result):
        if len(self.dense_fws) == 0 or len(self.dense_bws) == 0:
            logger.info(f"config {self.config} parse profile result is empty ")
            return
        profile_result[4] = self.dense_bw/self.dense_fw
        profile_result[7] = self.dense_fw
        pass