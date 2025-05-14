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

import math
from utils.logger import logger


class ExpertFilterManager:
    def __init__(self, mindformers_args, gbs):
        self.expert_filters = []
        self.mindformers_args = mindformers_args
        self.gbs = gbs

    @staticmethod
    def sequential_combination(selected_experiences, candidate_space):
        result = candidate_space
        for exp in selected_experiences:
            result = exp(result)
        return result

    @staticmethod
    def get_cp(config):
        return config[0][0][2]

    @staticmethod
    def get_dp(config):
        return config[0][0][0]

    @staticmethod
    def get_tp(config):
        return config[0][0][1]

    @staticmethod
    def get_pp(config):
        return config[0][0][3]

    @staticmethod
    def get_sp_switch(config):
        return config[1]

    @staticmethod
    def get_world_size(config):
        return math.prod(config[0][0])

    @staticmethod
    def get_mbs(config):
        return config[0][1][3]

    @staticmethod
    def get_ep(config):
        return config[0][1][0]

    def get_gbs(self):
        return self.gbs

    def get_num_layers(self):
        return self.mindformers_args.model.model_config.num_layers

    def get_mbn(self):
        return self.mindformers_args.parallel_config.micro_batch_num

    def add_experience(self, experience_function):
        """
        添加一个专家经验函数到列表中
        :param experience_function: 要添加的专家经验函数
        """
        self.expert_filters.append(experience_function)
        logger.info(f"add experience succ: {experience_function.__name__}")

    def remove_experience(self, experience_function):
        """
        从列表中移除一个专家经验函数
        :param experience_function: 要移除的专家经验函数
        """
        if experience_function in self.expert_filters:
            self.expert_filters.remove(experience_function)
            logger.info(f"remove experience succ: {experience_function.__name__}")
        else:
            logger.info(f"can not find experience: {experience_function.__name__}，无法移除。")

    def cp_for_deepseek_expert(self, candidate_space):
        # 2.28deepseek版本cp = 1
        return [config for config in candidate_space if self.get_cp(config) == 1]

    def pp_for_deepseek(self, candidate_space):
        # 万卡训练deepseek, 要求pp>1, pp过小内存会超
        return [config for config in candidate_space if self.get_pp(config) > 1]

    def pp_for_768die(self, candidate_space):
        # 768die, 要求pp<=32,
        return [config for config in candidate_space if self.get_pp(config) <= 32]

    def tp_for_910b_expert(self, candidate_space):
        # 910b为1机8卡，机间通信耗时过大，因此不能超过8
        return [config for config in candidate_space if self.get_tp(config) <= 8]

    def tp_for_910c_expert(self, candidate_space):
        # 910c超节点技术，专家经验不超过64即可from xiaoshihan
        return [config for config in candidate_space if self.get_tp(config) <= 64]


    def tp_for_910c_768die(self, candidate_space):
        # 910c超节点技术，768die, tp要是2的幂
        return [config for config in candidate_space if self.get_tp(config) % 3 != 0]

    def tp_for_yoco_expert(self, candidate_space):
        return [config for config in candidate_space if 56 % self.get_tp(config) == 0]

    def ep_for_910c_expert(self, candidate_space):
        return [config for config in candidate_space if self.get_ep(config) <= 64]

    def sp_for_lm_expert(self, candidate_space):
        # 在千卡规模及以上sp一定开启，千卡规模以下可支持sp搜索
        world_size = self.get_world_size(candidate_space[0])
        return [config for config in candidate_space
                if world_size < 1000 or self.get_sp_switch(config)]

    def pp_for_mbs_expert(self, candidate_space):
        return [config for config in candidate_space if
                self.get_pp(config) <=
                min(self.get_num_layers(), self.get_gbs() // self.get_dp(config) // self.get_mbs(config))]

    def gbs_for_dp_expert(self, candidate_space):
        return [config for config in candidate_space if self.get_gbs() % self.get_dp(config) == 0]

def select_profile_config(valid_configs):
    # 获取config中unique part1 + [max-ep, max-op, 1, 1] + true
    part1 = set()
    for config in valid_configs:
        part1.add(config[0][0])
    ret = []
    for config_par1 in part1:
        part2 = (1, 1, 1, 1)
        ret.append([[config_par1, part2], True])
    return ret

def expert_filter_configs(search_spaces, mindformers_args, gbs):
    """

    :param search_spaces: 初始搜索空间 [[(dp, tp, cp, pp), (ep, op, vp, mbs)], sp]
    :param mindformers_args: yaml配置信息
    :return: 使用专家经验剪枝搜索空间后得到的配置 [[(dp, tp, cp, pp), (ep, op, vp, mbs)], sp]
    """
    expert_manager = ExpertFilterManager(mindformers_args, gbs)
    expert_manager.add_experience(expert_manager.cp_for_deepseek_expert)
    expert_manager.add_experience(expert_manager.tp_for_910c_expert)
    expert_manager.add_experience(expert_manager.ep_for_910c_expert)
    expert_manager.add_experience(expert_manager.sp_for_lm_expert)
    expert_manager.add_experience(expert_manager.pp_for_mbs_expert)
    expert_manager.add_experience(expert_manager.gbs_for_dp_expert)
    expert_manager.add_experience(expert_manager.pp_for_deepseek)
    # add for 768die
    expert_manager.add_experience(expert_manager.pp_for_768die)
    expert_manager.add_experience(expert_manager.tp_for_910c_768die)
    # add for yoco model
    expert_manager.add_experience(expert_manager.tp_for_yoco_expert)
    valid_configs = expert_manager.sequential_combination(expert_manager.expert_filters, search_spaces)
    return valid_configs