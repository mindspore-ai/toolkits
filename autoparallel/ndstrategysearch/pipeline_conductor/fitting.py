import numpy as np
from utils.logger import logger
from pipeline_conductor import solution
from pipeline_conductor import dryrun
from pipeline_conductor import pp_util

import os


class FitMem:
    cur_peak_mem = []

    def __init__(self, cur_solution: solution.Solution):
        self.cur_solution = cur_solution
        self.cur_peak_mem = cur_solution.check_peak_mem
        self.peaks = cur_solution.peak_num
        self.init_config = cur_solution.init_config
        self.init_memory = cur_solution.init_config.memory

    def linear_fit(self):
        if self.init_config.expert_input.is_select_recomp and self.init_config.expert_input.is_full_recomp:
            length_x_lim = 11
        elif not self.init_config.expert_input.is_select_recomp and not self.init_config.expert_input.is_full_recomp:
            length_x_lim = 7
        else:
            length_x_lim = 9
        if self.init_config.pipeline_stage < length_x_lim:
            length_x = self.init_config.pipeline_stage
        else:
            length_x = length_x_lim
        if length_x < 4:
            logger.warning(f'can not fit for pipeline_stage = {self.init_config.pipeline_stage}')
            return
        coe_a, array_b = self.form_coe_matrix_mem_array(length_x)
        np.set_printoptions(suppress=True, precision=3)  # 禁用科学计数法，保留 6 位小数
        modified_mem, res, rank, s = np.linalg.lstsq(coe_a, array_b, rcond=None)
        logger.info(f'the residual = {np.linalg.norm(res)}')
        logger.info(f'the rank = {rank}')
        if np.linalg.norm(res) > 1e-3 or rank < self.init_config.pipeline_stage:
            logger.warning(f'The distribution can not correct the memory!')
        else:
            modified_mem = list(np.round(np.array(modified_mem), decimals=1))
            self.correct_mem(modified_mem)

    def correct_mem(self, modify_mem):
        self.init_memory.static_mem0 = modify_mem[0]
        self.init_memory.static_mem = modify_mem[1]
        self.init_memory.lm_head_mem = modify_mem[2]
        self.init_memory.act_mem = modify_mem[3]

        if len(modify_mem) == 5:
            self.init_memory.layer_mem = modify_mem[4]
        if len(modify_mem) == 6:
            self.init_memory.layer_mem = modify_mem[4]
            self.init_memory.re_comp_mem = modify_mem[5]
        if len(modify_mem) == 8:
            self.init_memory.layer_mem = modify_mem[4]
            self.init_memory.re_comp_mem = modify_mem[5]
            self.init_memory.layer_mem012 = modify_mem[6]
            self.init_memory.act_mem12 = modify_mem[7]
            self.init_memory.act_mem0 = modify_mem[7]
        if self.init_config.expert_input.is_select_recomp:
            if len(modify_mem) == 9:
                self.init_memory.layer_mem = modify_mem[4]
                self.init_memory.re_comp_mem = modify_mem[5]
                self.init_memory.layer_mem012 = modify_mem[6]
                self.init_memory.act_mem12 = modify_mem[7]
                self.init_memory.re_comp_mem12 = modify_mem[8]
                self.init_memory.re_comp_mem0 = modify_mem[8]
            if len(modify_mem) == 10:
                self.init_memory.layer_mem = modify_mem[4]
                self.init_memory.re_comp_mem = modify_mem[5]
                self.init_memory.layer_mem012 = modify_mem[6]
                self.init_memory.act_mem12 = modify_mem[7]
                self.init_memory.re_comp_mem12 = modify_mem[8]
                self.init_memory.re_comp_mem0 = modify_mem[8]
                self.init_memory.select_mem = modify_mem[9]
            if len(modify_mem) > 10:
                self.init_memory.layer_mem = modify_mem[4]
                self.init_memory.re_comp_mem = modify_mem[5]
                self.init_memory.layer_mem012 = modify_mem[6]
                self.init_memory.act_mem12 = modify_mem[7]
                self.init_memory.re_comp_mem12 = modify_mem[8]
                self.init_memory.re_comp_mem0 = modify_mem[8]
                self.init_memory.select_mem = modify_mem[9]
                self.init_memory.select_mem12 = modify_mem[10]
                self.init_memory.select_mem0 = self.init_memory.select_mem12
        else:
            if len(modify_mem) >= 9:
                self.init_memory.layer_mem = modify_mem[4]
                self.init_memory.re_comp_mem = modify_mem[5]
                self.init_memory.layer_mem012 = modify_mem[6]
                self.init_memory.act_mem12 = modify_mem[7]
                self.init_memory.re_comp_mem12 = modify_mem[8]
                self.init_memory.re_comp_mem0 = modify_mem[8]
        self.init_memory.update_up_mem()
        logger.info('The correct memory information:')
        self.init_memory.print_mem()

    def form_coe_matrix_mem_array(self, length_x):
        coe_a = np.empty((self.init_config.pipeline_stage, length_x), float)
        array_b = np.empty(self.init_config.pipeline_stage, float)
        for stage in range(self.init_config.pipeline_stage):
            if stage == 0:
                coe_a[stage][0] = 1
                coe_a[stage][1] = 0
                coe_a[stage][2] = 0
            elif stage == self.init_config.pipeline_stage - 1:
                coe_a[stage][0] = 0
                coe_a[stage][1] = 0
                coe_a[stage][2] = 1
            else:
                coe_a[stage][0] = 0
                coe_a[stage][1] = 1
                coe_a[stage][2] = 0
            coe_a[stage][3] = self.cur_solution.peak_num.peak_num_act_type2[stage]
            if length_x == 4:
                array_b[stage] = (self.cur_peak_mem[stage] - self.cur_solution.layer2_dis_stage[stage] *
                                  self.init_memory.layer_mem -
                                  self.peaks.peak_num_recompute_type2[stage] * self.init_memory.re_comp_mem  -
                                  self.cur_solution.layer1_dis_stage[stage] * self.init_memory.layer_mem012 -
                                  self.peaks.peak_num_act_type1[stage] * self.init_memory.act_mem12 -
                                  self.peaks.peak_num_recompute_type1[stage] * self.init_memory.re_comp_mem12 -
                                  self.peaks.peak_num_select_recom_type2[stage] * self.init_memory.select_mem -
                                  self.peaks.peak_num_select_recom_type1 * self.init_memory.select_mem12)
                continue
            if length_x == 6:
                coe_a[stage][4] = self.cur_solution.layer2_dis_stage[stage]
                coe_a[stage][5] = self.peaks.peak_num_recompute_type2[stage]
                array_b[stage] = (self.cur_peak_mem[stage]  -
                                  self.cur_solution.layer1_dis_stage[stage] * self.init_memory.layer_mem012 -
                                  self.peaks.peak_num_act_type1[stage] * self.init_memory.act_mem12 -
                                  self.peaks.peak_num_recompute_type1[stage] * self.init_memory.re_comp_mem12 -
                                  self.peaks.peak_num_select_recom_type2[stage] * self.init_memory.select_mem -
                                  self.peaks.peak_num_select_recom_type1 * self.init_memory.select_mem12)
                continue
            if length_x == 8:
                coe_a[stage][4] = self.cur_solution.layer2_dis_stage[stage]
                coe_a[stage][5] = self.peaks.peak_num_recompute_type2[stage]
                coe_a[stage][6] = self.cur_solution.layer1_dis_stage[stage]
                coe_a[stage][7] = self.peaks.peak_num_act_type1[stage]
                array_b[stage] = (self.cur_peak_mem[stage] - self.peaks.peak_num_recompute_type1[stage] *
                                  self.init_memory.re_comp_mem12 - self.peaks.peak_num_select_recom_type2[stage] *
                                  self.init_memory.select_mem -
                                  self.peaks.peak_num_select_recom_type1[stage] * self.init_memory.select_mem12)
                continue
            if length_x == 9:
                coe_a[stage][4] = self.cur_solution.layer2_dis_stage[stage]
                coe_a[stage][5] = self.peaks.peak_num_recompute_type2[stage]
                coe_a[stage][6] = self.cur_solution.layer1_dis_stage[stage]
                coe_a[stage][7] = self.peaks.peak_num_act_type1[stage]
                coe_a[stage][8] = self.peaks.peak_num_recompute_type1[stage]
                array_b[stage] = (self.cur_peak_mem[stage] - self.peaks.peak_num_select_recom_type2[stage] *
                                  self.init_memory.select_mem -
                                  self.peaks.peak_num_select_recom_type1 * self.init_memory.select_mem12)
                continue
            if self.init_config.expert_input.is_select_recomp:
                if length_x == 10:
                    coe_a[stage][4] = self.cur_solution.layer2_dis_stage[stage]
                    coe_a[stage][5] = self.peaks.peak_num_recompute_type2[stage]
                    coe_a[stage][6] = self.cur_solution.layer1_dis_stage[stage]
                    coe_a[stage][7] = self.peaks.peak_num_act_type1[stage]
                    coe_a[stage][8] = self.peaks.peak_num_recompute_type1[stage]
                    coe_a[stage][9] = self.peaks.peak_num_select_recom_type2[stage]
                    array_b[stage] = (self.cur_peak_mem[stage] -
                                      self.peaks.peak_num_select_recom_type1 * self.init_memory.select_mem12)
                    continue
                if length_x >= 11:
                    coe_a[stage][4] = self.cur_solution.layer2_dis_stage[stage]
                    coe_a[stage][5] = self.peaks.peak_num_recompute_type2[stage]
                    coe_a[stage][6] = self.cur_solution.layer1_dis_stage[stage]
                    coe_a[stage][7] = self.peaks.peak_num_act_type1[stage]
                    coe_a[stage][8] = self.peaks.peak_num_recompute_type1[stage]
                    coe_a[stage][9] = self.peaks.peak_num_select_recom_type2[stage]
                    coe_a[stage][10] = self.peaks.peak_num_select_recom_type1[stage]
                    array_b[stage] = self.cur_peak_mem[stage]
                    continue
            else:
                if length_x > 9:
                    coe_a[stage][4] = self.cur_solution.layer2_dis_stage[stage]
                    coe_a[stage][5] = self.peaks.peak_num_recompute_type2[stage]
                    coe_a[stage][6] = self.cur_solution.layer1_dis_stage[stage]
                    coe_a[stage][7] = self.peaks.peak_num_act_type1[stage]
                    coe_a[stage][8] = self.peaks.peak_num_recompute_type1[stage]
                    array_b[stage] = self.cur_peak_mem[stage]
                    continue
        return coe_a, array_b

    def is_over_mem(self):
        over_mem = self.get_over_mem(self.cur_solution)
        if all(over_mem[stage] == 0 for stage in range(len(over_mem))):
            recompute_config = pp_util.build_recompute_config(True, True, self.cur_solution.
                                                              rs_dis.tolist(), self.cur_solution.ra_dis.tolist())
            pp_util.write_config_to_yaml(recompute_config, self.cur_solution.offset.tolist(), self.init_config.
                                         yaml_file)
            logger.info(f'The result is available for training, config has write to '
                        f'{self.init_config.yaml_file}!')
            return over_mem, False
        else:
            return over_mem, True

    def reduce_mem_lim_for_fitting(self, over_mem, i):
        if over_mem.keys().__contains__(0):
            self.init_config.memory.mem_lim_stage0 -= over_mem[0] * (i + 1)
        if over_mem.keys().__contains__(self.init_config.pipeline_stage - 1):
            self.init_config.memory.mem_lim_last -= over_mem[self.init_config.pipeline_stage - 1] * (i + 1)
        over_mem = {key: value for key, value in over_mem.items() if key != 0 and
                    key != self.init_config.pipeline_stage - 1}
        if over_mem:
            self.init_config.memory.mem_lim_others -= max(over_mem.values()) * (i + 1)

    def set_peak_mem_by_dryrun(self, cur_solution: solution.Solution):
        init_config = cur_solution.init_config
        expert_input = init_config.expert_input
        dry_run = dryrun.DryRun(expert_input.yaml_file, expert_input.mind_former_file,
                                expert_input.double_check_dryrun_filename)
        recompute_config = pp_util.build_recompute_config(True, True, cur_solution.rs_dis.
                                                          tolist(), cur_solution.ra_dis.tolist())
        total_number = init_config.num_layers_type1 + init_config.num_layers_type2
        dry_run.start_dryrun(recompute_config, cur_solution.offset.tolist(), total_number,
                             init_config.pp_interleave_num,
                             init_config.pipeline_stage, init_config.rank_size, init_config.num_layers_type1,
                             init_config.micro_batch_num)
        cur_solution.check_peak_mem = dry_run.extract_memory_info_act(init_config.pipeline_stage)
        self.cur_peak_mem = cur_solution.check_peak_mem
        logger.info(f'check peak_mem = {cur_solution.check_peak_mem}')

    def get_over_mem(self, cur_solution: solution.Solution):
        self.set_peak_mem_by_dryrun(cur_solution)
        over_mem = {}
        init_config = cur_solution.init_config
        for stage in range(init_config.pipeline_stage):
            if cur_solution.check_peak_mem[stage] - init_config.mem_lim > 0:
                logger.warning(f'The stage{stage} result is over memory limit! please check the offset!')
                over_mem[stage] = cur_solution.check_peak_mem[stage] - init_config.mem_lim
            else:
                over_mem[stage] = 0
        return over_mem

