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

import re

import numpy as np
import sys
from ortools.linear_solver import pywraplp
from utils.logger import logger

from pipeline_conductor import pp_util
from pipeline_conductor.start_service import InitConfig, HIGHS_NAME
from pipeline_conductor import micro


class Model:
    def __init__(self, model_input: InitConfig):
        # 初始化输入
        self.input = model_input
        self.memory = model_input.memory
        self.expert_input = model_input.expert_input
        # 前后计算层数、时间
        self.x_type1 = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        self.x_type2 = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        self.offset = None
        self.f_duration = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        self.b_duration = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        # 当前stage，interleave是否存在某一个种type的层
        self.indicator_type1 = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        self.indicator_type2 = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        # 选择重计算、完全重计算、内存
        self.rs_type1 = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        self.rs_type2 = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        self.ra_type1 = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        self.ra_type2 = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        self.rs_or_ra = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        self.mem = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num), dtype=object)
        # 前后向开始时间、micro batch在各个part中分布
        magic_number = 5
        if self.input.micro_batch_num // self.input.pipeline_stage >= magic_number + 1:
            dummy_mbn = self.input.micro_batch_num % self.input.pipeline_stage + magic_number * self.input.pipeline_stage
            self.residue = self.input.micro_batch_num // self.input.pipeline_stage - magic_number
            temp_parts = magic_number
            self.input.parts = magic_number
        else:
            dummy_mbn = self.input.micro_batch_num
            self.residue = 0
            temp_parts = model_input.parts
            self.input.parts = temp_parts
        self.forward_s = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num, self.input.parts,
                                   dummy_mbn, self.input.seq_splits), dtype=object)
        self.backward_s = np.empty((self.input.pipeline_stage, self.input.pp_interleave_num, self.input.parts,
                                    dummy_mbn, self.input.seq_splits), dtype=object)
        self.distribution = pp_util.construct_distribution(dummy_mbn, self.input.pipeline_stage)
        # 形成前后向任务列表
        self.sort_micro = micro.SortMicro(temp_parts, model_input.pp_interleave_num, model_input.pipeline_stage,
                                     self.distribution, self.expert_input.low_mem, model_input.seq_splits)
        self.final_orders = self.sort_micro.final_orders
        # 求解初始化
        self.solver = pywraplp.Solver.CreateSolver(model_input.expert_input.solver_name)
        if not self.solver:
            self.solver = pywraplp.Solver.CreateSolver(HIGHS_NAME)
        self.layer_upper = 20
        self.b_duration_upper = self.layer_upper * 3
        self.mem_upper = 60000
        self.time_upper = 99999
        self.min_time = None
        self.gap = None
        self.model_status = None
        self.solution_status = None
        self.run_time = None

    def define_variables(self):
        self.stable_dur = self.solver.NumVar(0, self.time_upper, f'stable_duration')
        for stage in range(self.input.pipeline_stage):
            for vpp in range(self.input.pp_interleave_num):
                self.x_type1[stage][vpp] = self.solver.IntVar(0, self.layer_upper, f'x_type1_{stage}_{vpp}')
                self.x_type2[stage][vpp] = self.solver.IntVar(0, self.layer_upper, f'x_type2_{stage}_{vpp}')
                self.indicator_type1[stage][vpp] = self.solver.BoolVar(f'indicator1_{stage}_{vpp}')
                self.indicator_type2[stage][vpp] = self.solver.BoolVar(f'indicator2_{stage}_{vpp}')
                self.f_duration[stage][vpp] = self.solver.NumVar(0, self.layer_upper, f'f_duration_{stage}_{vpp}')
                self.b_duration[stage][vpp] = self.solver.NumVar(0, self.b_duration_upper, f'b_duration_{stage}_{vpp}')
                if self.input.expert_input.is_select_recomp:
                    self.rs_type1[stage][vpp] = self.solver.IntVar(
                        0, self.layer_upper, f'rs_type1_{stage}_{vpp}')
                    self.rs_type2[stage][vpp] = self.solver.IntVar(
                        0, self.layer_upper, f'rs_type2_{stage}_{vpp}')
                else:
                    self.rs_type1[stage][vpp] = self.solver.IntVar(0, 0, f'rs_type1_{stage}_{vpp}')
                    self.rs_type2[stage][vpp] = self.solver.IntVar(0, 0, f'rs_type2_{stage}_{vpp}')
                if self.input.expert_input.is_full_recomp:
                    self.ra_type1[stage][vpp] = self.solver.IntVar(0, self.layer_upper, f'ra_type1_{stage}_{vpp}')
                    self.ra_type2[stage][vpp] = self.solver.IntVar(0, self.layer_upper, f'ra_type2_{stage}_{vpp}')
                else:
                    self.ra_type1[stage][vpp] = self.solver.IntVar(0, 0, f'ra_type1_{stage}_{vpp}')
                    self.ra_type2[stage][vpp] = self.solver.IntVar(0, 0, f'ra_type2_{stage}_{vpp}')
                if not self.expert_input.is_support_ra_with_rs:
                    self.rs_or_ra[stage][vpp] = self.solver.IntVar(0, 1, f'rs_or_ra_{stage}_{vpp}')
                self.mem[stage][vpp] = self.solver.NumVar(0, self.mem_upper, f'mem_{stage}_{vpp}')

        for stage in range(self.input.pipeline_stage):
            for vpp in range(self.input.pp_interleave_num):
                for part in range(self.input.parts):
                    for micro_id in range(self.distribution[part]):
                        for split in range(self.input.seq_splits):
                            self.forward_s[stage][vpp][part][micro_id][split] = (
                                self.solver.NumVar(0, self.time_upper,
                                                   f'forward_s_{stage}_{vpp}_{part}_{micro_id}_{split}'))
                            self.backward_s[stage][vpp][part][micro_id][split] = (
                                self.solver.NumVar(0, self.time_upper,
                                                   f'backward_s{stage}_{vpp}_{part}_{micro_id}_{split}'))
        logger.info(f'Number of variables = {self.solver.NumVariables()}')

    def define_constraint(self):
        # 先type1后type2的层分配约束
        for vpp in range(self.input.pp_interleave_num):
            for stage in range(self.input.pipeline_stage):
                self.solver.Add(self.x_type1[stage][vpp] <= self.layer_upper * self.indicator_type1[stage][vpp])
                self.solver.Add(self.x_type1[stage][vpp] >= self.indicator_type1[stage][vpp])
                self.solver.Add(self.x_type2[stage][vpp] <= self.layer_upper * self.indicator_type2[stage][vpp])
                self.solver.Add(self.x_type2[stage][vpp] >= self.indicator_type2[stage][vpp])
                if stage < self.input.pipeline_stage-1:
                    self.solver.Add(self.indicator_type1[stage][vpp] >= self.indicator_type1[stage+1][vpp])
                    self.solver.Add(self.indicator_type2[stage][vpp] <= self.indicator_type2[stage+1][vpp])
            if vpp < self.input.pp_interleave_num - 1:
                self.solver.Add(self.indicator_type1[self.input.pipeline_stage-1][vpp] >= self.indicator_type1[0][vpp + 1])
                self.solver.Add(self.indicator_type2[self.input.pipeline_stage-1][vpp] <= self.indicator_type2[0][vpp + 1])

        # 前后向计算时间约束
        if self.input.expert_input.is_head_loss_input:
            head_loss = self.input.expert_input.head_loss
        else:
            head_loss = ((self.input.vocab_size / 2 / self.input.hidden_size) /
                         (1 + 1 + 3 * self.input.intermediate_size / 2 / self.input.hidden_size + self.input.seq_length
                          / self.input.hidden_size) * 1.6)

        for stage in range(self.input.pipeline_stage):
            for vpp in range(self.input.pp_interleave_num):
                # self.expert_input.layer_ratio is an integer now (used to be a list of integers)
                if stage == self.input.pipeline_stage - 1 and vpp == self.input.pp_interleave_num - 1:
                    self.solver.Add(self.f_duration[stage][vpp] == self.x_type2[stage][vpp] +
                                self.x_type1[stage][vpp]*self.expert_input.layer_ratio + head_loss)
                else:
                    self.solver.Add(self.f_duration[stage][vpp] == self.x_type2[stage][vpp] +
                                self.x_type1[stage][vpp] * self.expert_input.layer_ratio)
                self.solver.Add(self.b_duration[stage][vpp] == self.expert_input.backward_ratio *
                                self.f_duration[stage][vpp] + self.expert_input.srRatio *
                                (self.rs_type2[stage][vpp] + self.rs_type1[stage][vpp] * self.expert_input.layer_ratio)
                                + self.expert_input.recompute_ratio * self.ra_type2[stage][vpp] +
                                self.ra_type1[stage][vpp] * self.expert_input.layer_ratio)

        # 同stage micro-batch之间的约束
        for stage in range(self.input.pipeline_stage):
            stage_order = self.final_orders[stage]
            for i in range(len(stage_order) - 1):
                p0, vpp0, state0, id0, split0 = (stage_order[i].part, stage_order[i].vpp, stage_order[i].state,
                                                 stage_order[i].micro_id, stage_order[i].split)
                p1, vpp1, state1, id1, split1 = (stage_order[i + 1].part, stage_order[i + 1].vpp,
                                                 stage_order[i + 1].state, stage_order[i + 1].micro_id,
                                                 stage_order[i].split)
                if state0 == 'f':
                    if state1 == 'f':
                        self.solver.Add(self.forward_s[stage][vpp0][p0][id0][split0] + self.f_duration[stage][vpp0] /
                                        self.input.seq_splits <= self.forward_s[stage][vpp1][p1][id1][split1])
                    else:
                        self.solver.Add(self.forward_s[stage][vpp0][p0][id0][split0] + self.f_duration[stage][vpp0] /
                                        self.input.seq_splits <= self.backward_s[stage][vpp1][p1][id1][split1])
                else:
                    if state1 == 'f':
                        self.solver.Add(self.backward_s[stage][vpp0][p0][id0][split0] + self.b_duration[stage][vpp0] /
                                        self.input.seq_splits <= self.forward_s[stage][vpp1][p1][id1][split1])
                    else:
                        self.solver.Add(self.backward_s[stage][vpp0][p0][id0][split0] + self.b_duration[stage][vpp0] /
                                        self.input.seq_splits <= self.backward_s[stage][vpp1][p1][id1][split1])
        # 同micro-batch，stage间的约束
        for part in range(self.input.parts):
            for micro_id in range(self.distribution[part]):
                for split in range(self.input.seq_splits):
                    for vpp in range(self.input.pp_interleave_num):
                        for stage in range(self.input.pipeline_stage):
                            # 前向：
                            if stage != self.input.pipeline_stage - 1:
                                self.solver.Add(
                                    self.forward_s[stage][vpp][part][micro_id][split] + self.f_duration[stage][vpp] /
                                    self.input.seq_splits <= self.forward_s[stage + 1][vpp][part][micro_id][split])
                            elif vpp != self.input.pp_interleave_num - 1:
                                self.solver.Add(
                                    self.forward_s[stage][vpp][part][micro_id][split] + self.f_duration[stage][vpp] /
                                    self.input.seq_splits <= self.forward_s[0][vpp + 1][part][micro_id][split])
                            else:
                                self.solver.Add(
                                    self.forward_s[stage][vpp][part][micro_id][split] + self.f_duration[stage][vpp] /
                                    self.input.seq_splits <= self.backward_s[stage][vpp][part][micro_id][split])
                            # 后向：
                            if stage != 0:
                                self.solver.Add(
                                    self.backward_s[stage][vpp][part][micro_id][split] + self.b_duration[stage][vpp] /
                                    self.input.seq_splits <= self.backward_s[stage - 1][vpp][part][micro_id][split])
                            else:
                                if vpp != 0:
                                    self.solver.Add(
                                        self.backward_s[stage][vpp][part][micro_id][split] + self.b_duration[stage][vpp]
                                        / self.input.seq_splits <= self.backward_s[
                                            self.input.pipeline_stage - 1][vpp - 1][part][micro_id][split])

        # 内存约束
        for stage in range(self.input.pipeline_stage):
            for vpp in range(self.input.pp_interleave_num):
                self.solver.Add(self.mem[stage][vpp] == self.memory.act_mem *
                                (self.x_type2[stage][vpp] - self.rs_type2[stage][vpp] - self.ra_type2[stage][vpp]) +
                                self.memory.select_mem * self.rs_type2[stage][vpp] +
                                self.memory.re_comp_mem * self.ra_type2[stage][vpp] +
                                self.memory.act_mem0 * (self.x_type1[stage][vpp] - self.rs_type1[stage][vpp] -
                                                        self.ra_type1[stage][vpp]) + self.memory.select_mem0 *
                                self.rs_type1[stage][vpp] + self.memory.re_comp_mem0 * self.ra_type1[stage][vpp])

        for stage in range(self.input.pipeline_stage):
            total_layer_mem = 0
            for vpp in range(self.input.pp_interleave_num):
                total_layer_mem += (self.x_type1[stage][vpp] * self.memory.layer_mem012 + self.x_type2[stage][vpp] *
                                    self.memory.layer_mem)
            for sub in range(1, len(self.final_orders[stage]) + 1):
                consume_mem = total_layer_mem
                for i in range(sub):
                    part, vpp, state, micro_id, split = (self.final_orders[stage][i].part,
                                                         self.final_orders[stage][i].vpp,
                                                         self.final_orders[stage][i].state,
                                                         self.final_orders[stage][i].micro_id,
                                                         self.final_orders[stage][i].split)
                    # 计算时间均分，因此这里激活内存做均分处理
                    if state == 'f':
                        consume_mem += self.mem[stage][vpp] / self.input.seq_splits
                    else:
                        consume_mem -= self.mem[stage][vpp] / self.input.seq_splits
                if stage == 0:
                    self.solver.Add(consume_mem <= self.memory.mem_lim_stage0)
                elif stage == self.input.pipeline_stage - 1:
                    self.solver.Add(consume_mem <= self.memory.mem_lim_last)
                else:
                    self.solver.Add(consume_mem <= self.memory.mem_lim_others)
        # layer约束
        layers_type1 = 0
        layers_type2 = 0
        indicator_total = 0
        for stage in range(self.input.pipeline_stage):
            for vpp in range(self.input.pp_interleave_num):
                layers_type1 += self.x_type1[stage][vpp]
                layers_type2 += self.x_type2[stage][vpp]
                indicator_total += self.indicator_type1[stage][vpp] + self.indicator_type2[stage][vpp]
                self.solver.Add(self.x_type1[stage][vpp] >= self.ra_type1[stage][vpp] + self.rs_type1[stage][vpp])
                self.solver.Add(self.x_type2[stage][vpp] >= self.ra_type2[stage][vpp] + self.rs_type2[stage][vpp])
                if not self.expert_input.is_support_ra_with_rs:
                    self.solver.Add(self.rs_type1[stage][vpp] <= self.layer_upper * self.rs_or_ra[stage][vpp])
                    self.solver.Add(self.rs_type2[stage][vpp] <= self.layer_upper * self.rs_or_ra[stage][vpp])
                    self.solver.Add(self.ra_type1[stage][vpp] <= self.layer_upper * (1 - self.rs_or_ra[stage][vpp]))
                    self.solver.Add(self.ra_type2[stage][vpp] <= self.layer_upper * (1 - self.rs_or_ra[stage][vpp]))
        self.solver.Add(layers_type1 == self.input.num_layers_type1)
        self.solver.Add(layers_type2 == self.input.num_layers_type2)
        self.solver.Add(indicator_total >= self.input.pipeline_stage * self.input.pp_interleave_num)
        self.solver.Add(indicator_total <= self.input.pipeline_stage * self.input.pp_interleave_num + 1)
        for s in range(self.input.pipeline_stage):
            self.solver.Add(self.stable_dur >= self.forward_s[s][0][-1][0][0] - self.forward_s[s][0][-2][0][0])
        logger.info(f"Number of constraints = {self.solver.NumConstraints()}")

    def define_obj(self):
        self.solver.Minimize(self.backward_s[0][0][-1][self.distribution[-1] - 1][0] + self.b_duration[0][0] /
                             self.input.seq_splits + self.residue * self.stable_dur)

    def output_model(self, mps_file):
        with open(mps_file, 'w') as file:
            mps_text = self.solver.ExportModelAsMpsFormat(False, False)
            file.write(mps_text)
        logger.info(f'Had write to file: {mps_file}')

    def solve(self):
        self.solver.EnableOutput()
        # self.solver.SetSolverSpecificParametersAsString('output_flag=1')
        logger.info(f'Solving with {self.solver.SolverVersion()}')
        if self.expert_input.time_limit != sys.maxsize:
            self.solver.SetTimeLimit(self.expert_input.time_limit)
        self.solver.Solve()
        self.min_time = self.solver.Objective().Value()
        logger.info(f'The objective value = {self.min_time}')


if __name__ == '__main__':
    yaml_file = 'C:\\working\\768_4k.yaml'
