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

import itertools
import math

from utils.common import cal_model_layers_num, cal_world_size


def cal_factor(num):
    factors = []
    for i in range(1, num + 1):
        if num % i == 0:
            factors.append(i)
    return factors

def find_integers_less_than_ceil(num_layers, pp):
    ceil_value = math.ceil(num_layers / pp)
    result = list(range(ceil_value))
    return result

def find_ep(dp, tp, expert_num):
    product = tp * dp
    # 用于存储满足条件的 ep 值的列表
    ep_list = []
    # 遍历从 1 到 product 的所有数
    for i in range(1, product + 1):
        if product % i == 0 & i <= expert_num:
            ep_list.append(i)
    return ep_list

def build_initial_spaces(input_args):
    world_size = cal_world_size(input_args)
    # part1. 求[dp, tp, cp, pp]的取值范围
    # 找到 world_size 的所有因子
    factors = cal_factor(world_size)

    # 找出所有可能的四个正整数乘积等于world_size
    part1_combinations = []
    for combination in itertools.combinations_with_replacement(factors, 4):
        product = 1
        for num in combination:
            product *= num
        if product == world_size:
            perms = list(itertools.permutations(combination))
            unique_configs = list(set(perms))
            part1_combinations.extend(unique_configs)

    # part2. [ep, op, vp, mbs]
    # ep是dp*tp的约数
    # 优化器并行可以是dp的任意因子
    # vp是 < ceil(总层数/pp)的任意整数
    # mbs 取值(1， 2， 4)中
    part2_combinations = []
    num_layers = input_args.num_layers
    for world_size_config in part1_combinations:
        op_options = cal_factor(world_size_config[0])
        vp_options = find_integers_less_than_ceil(num_layers, world_size_config[3])
        mbs_options = [1, 2, 4]
        ep_options = find_ep(world_size_config[0], world_size_config[1], input_args.expert_num)
        result = list(itertools.product(ep_options, op_options, vp_options, mbs_options))

        dp = world_size_config[0]
        tp = world_size_config[1]
        mul_dptp = dp * tp
        for tmp_result in result:
            ep = tmp_result[0]
            op = tmp_result[1]
            outer_dp = mul_dptp / ep
            if op % outer_dp == 0:
                # 格式 [(dp, tp, cp, pp), (ep, op, vp, mbs)]
                part2_combinations.append([world_size_config, tmp_result])

    # part3. sp只有开关与否 格式 [[(dp, tp, cp, pp), (ep, op, vp, mbs)], sp]
    final_combinations = []
    for part2_config in part2_combinations:
        final_combinations.append([part2_config, True])
        final_combinations.append([part2_config, False])
    return final_combinations