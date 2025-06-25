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

from utils.logger import logger

encoding = 'utf-8'

def find_file_by_name(directory, filename):
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None

def median_mean(durs):
    median_durs = durs[len(durs)//4 : len(durs) - len(durs)//4]
    return round(statistics.median(median_durs)/1000, 3)

class ProfileExe:

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
    recomp_bws, moe_fws1, moe_fws2, moe_bws, dense_bws, totals = [], [], [], [], [], []
    moe_fw1, moe_fw2, moe_bw, recomp_bw, dense_fw, dense_bw, head_ratio, num_last_stage_layers, mtp = 0,0,0,0,0,0,0,0,0
    att_tid, grad_att_tid = 9, 11

    def __init__(self):
        pass

    def load_profile(self, file, rk):
        global encoding
        path = os.path.abspath('.\\profile_info') + '\\' + str(file)
        path = find_file_by_name(path + '\\rank_' + str(rk), 'trace_view.json')
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
            if line['pid'] == 3 and line['tid'] == self.att_tid and line['name'] == 'flash_attention-FlashAttention':
                atts.append(float(line['ts']))
            if line['pid'] == 3 and line['tid'] == self.grad_att_tid and line['name'] == 'Grad_FlashAttentionScore':
                grad_atts.append(float(line['ts']))
        logger.info(f'num of atts is {len(atts)}, num of grad_atts is {len(grad_atts)}')
        return atts, grad_atts

    def stage_anal(self, pp, stage): #assuming we profiled 2 steps w/ micro = 32
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
                        self.moe_fw1.append(chunk[i+1] - chunk[i])
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
        """
        解析profile的对外接口
        Args:
            config: str--并行配置, 例如dp8tp4pp4ep32
            ranks:  list--采集的各个stage的rank id列表

        """
        self.config = config
        pp = len(ranks)
        for i in range(pp):
            self.load_profile(config, ranks[i])
            self.stage_anal(pp, i)
            self.release()

    def data_display(self):
        logger.info(f'moe_fws1 are {self.moe_fws1}')
        logger.info(f'moe_fws2 are {self.moe_fws2}')
        logger.info(f'moe_bws is {self.moe_bws}')
        logger.info(f'recomp_bws is {self.recomp_bws}')
        logger.info(f'dense_fw is {self.dense_fw}')
        logger.info(f'dense_bw is {self.dense_bws}')
        logger.info(f'totals is {self.totals}')

    def refined_data(self):
        self.moe_fw1 = median_mean(self.moe_fws1)
        self.moe_fw2 = median_mean(self.moe_fws2)
        self.moe_bws = median_mean(self.moe_bws)
        self.recomp_bws = median_mean(self.recomp_bws)
        self.dense_fw = median_mean(self.dense_fw)
        self.dense_bw = median_mean(self.dense_bw)
        total = median_mean(self.totals)
        self.mtp = total - self.num_last_stage_layers * (self.moe_fw2 + self.moe_bws)
        logger.info(f'config is {self.config}, dense forward is {self.dense_fw}, dense backward is {self.dense_bw}, moe forward is {self.moe_fw1} or {self.moe_fw2}, moe backward is {self.moe_bws}, moe recomp backward is {self.recomp_bws}, lmhead and MTP is {self.mtp}')

