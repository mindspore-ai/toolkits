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
import os
import math
import csv
from collections import defaultdict
from utils.logger import logger
from utils.common import cal_model_layers_num, generate_files, initial_offset
from utils.dryrun_manage import launch_dryrun, read_dryrun_info


def memory_simple_prune(config, profile_info):
    memory_max = config.memory_max
    tmp_layer = config.num_layer // config.pp_size
    profile_memory = config.pp_size * tmp_layer * profile_info.act_mem_full_recomp \
        + profile_info.embedding_mem + tmp_layer * profile_info.static_layer_MoE
    coe_mem = 0.2
    if profile_memory * (1 + coe_mem) > memory_max:
        return False
    return True

def trans_format(dryrun_info_init, test_ep):
    config_mem_dict = defaultdict(lambda: [0, 0])
    for config in dryrun_info_init:
        key = tuple(config[:3])
        peak_mem = config[4]
        ep = config[3]
        if ep == test_ep[0]:
            index = 0
        elif ep == test_ep[1]:
            index = 1
        else:
            logger.error(f"error ep {ep} is not supported")
            continue
        config_mem_dict[key][index] = peak_mem

    return [list(key) + value for key, value in config_mem_dict.items()]

def grey_box_memory_prune(mindformers_args, dryrun_info_init, test_ep, max_expert_parallel):
    """
    ep灰盒剪枝

    :param max_expert_parallel: ep最大值
    :param test_ep:
    :param dryrun_info_init: [dp, tp, pp, ep, peak_value]
    :param mindformers_args:
    :return: [dp, tp, pp, ep, evaluate_peak_mem]
    """
    dryrun_info = trans_format(dryrun_info_init, test_ep)
    ep1, ep2 = test_ep
    logger.info(f"dryrun_info len: {len(dryrun_info)} format [dp, tp, pp, peak_mem_ep{ep1}, peak_mem_ep{ep2}]")
    logger.info("\n".join(str(config) for config in dryrun_info))

    try:
        max_mem = int(re.search(r'\d+', mindformers_args.context.max_device_memory).group()) * 1024
    except (AttributeError, ValueError):
        max_mem = 58 * 1024

    memory_aware_configs = []
    logger.info("format: dp_tp_pp_ep_evaluateMem")
    ep_power = find_power_of_two(max_expert_parallel)
    for dp, tp, pp, peak_ep, peak_ep_double in dryrun_info:
        # 线性拟合ep会影响到的内存和ep不会影响到的内存
        ep_memory = (peak_ep - peak_ep_double) * test_ep[1] #所有专家的内存
        base_memory = peak_ep - ep_memory / test_ep[0]
        # 确定ep最大能开多大,最大为6, ep最大64
        ep_upperbound = 0
        for i in range(ep_power+1):
            if (dp*tp) % (2**i) == 0:
                ep_upperbound += 1
        # 输出满足内存上限的ep，如果ep64都不够就不返回
        for j in range(ep_upperbound):
            ep = 2 ** j
            evaluate_mem = base_memory + ep_memory / ep
            logger.info(f"{dp}_{tp}_{pp}_{ep}_{evaluate_mem}")
            if evaluate_mem <= max_mem:
                memory_aware_configs.append([dp, tp, pp, ep, evaluate_mem])
    return memory_aware_configs

def find_power_of_two(m):
    if m <= 0:
        return None
    power = math.log2(m)
    if power.is_integer():
        return int(power)
    return None

def filter_oom_by_dryrun(search_space, mindformers_args, para):
    # 生成要做dryrun的配置
    care_part_configs = select_dry_config(search_space)
    test_ep = (8, 16)
    dry_config = generate_dry_config(care_part_configs, mindformers_args, test_ep)
    dryrun_exe_switch = False if para.DRYRUN_DATA_DIR else True
    if dryrun_exe_switch:
        logger.info("need auto dryrun process")
        dryrun_yaml_dir = os.path.abspath(para.OUTPUT_PATH) + os.sep + 'dryrun_yaml' + os.sep
        generate_files(dry_config, dryrun_yaml_dir, "dryrun", para)
        dryrun_data_dir = os.path.join(os.path.abspath(para.OUTPUT_PATH), "dryrun_output")
        launch_dryrun(mindformers_args, dryrun_yaml_dir, dryrun_data_dir, para)
    else:
        dryrun_data_dir = para.DRYRUN_DATA_DIR

    dryrun_info = read_dryrun_info(dryrun_data_dir)
    candidate_configs = grey_box_memory_prune(mindformers_args, dryrun_info, test_ep, para.MAX_EXPERT_PARALLEL)
    generate_csv(para.OUTPUT_PATH, candidate_configs)
    return candidate_configs

def select_dry_config(valid_configs):
    """

    :param valid_configs: [[(dp, tp, cp, pp), (ep, op, vp, mbs)], sp]
    :return: [[(dp, tp, cp, pp), (ep, op, vp, mbs)], sp]
             其中vp=1, mbs=1, sp=true  每个(dp, tp, cp, pp)，对应ep最大的配置列表
    """
    first = valid_configs[0][0][0]
    max_ep = valid_configs[0][0][1][0]
    op_with_ep = valid_configs[0][0][1][1]
    ans = []
    for config in valid_configs:
        current_first = config[0][0]
        current_ep = config[0][1][0]
        current_op = config[0][1][1]

        if current_first == first:
            if current_ep > max_ep:
                max_ep = current_ep
                op_with_ep = current_op
        else:
            ans.append([[first, [max_ep, op_with_ep, 1, 1]], True])
            first = current_first
            max_ep = current_ep
            op_with_ep = current_op
    # 添加最后一组数据
    ans.append([[first, [max_ep, op_with_ep, 1, 1]], True])
    logger.info(f"Dryrun candidate config size: {len(ans)}")
    return ans

def generate_csv(output_path, dryrun_config):
    # 表头
    headers = ['dp', 'tp', 'pp', 'ep', 'evaluate_mem']

    # 写入 CSV 文件
    try:
        csv_path = os.path.join(os.path.abspath(output_path), "nd_candidate_config.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 写入表头
            writer.writerow(headers)
            # 写入数据
            writer.writerows(dryrun_config)
        logger.info("CSV file generate succ.")
    except Exception as e:
        logger.info(f"write CSV file fail: {e}")

def generate_dry_config(care_part_configs, mindformers_args, test_ep):
    """

    :param test_ep: 要做dryrun的ep
    :param care_part_configs: [[(dp, tp, cp, pp), (ep, op, vp, mbs)], sp]
    :param mindformers_args:
    :return: [dp, tp, pp, ep, offset] 其中ep = {4, 8}
    """
    dry_run_config = []
    layers_num = cal_model_layers_num(mindformers_args)
    for config in care_part_configs:
        dp, tp, _, pp = config[0][0]
        for ep in test_ep:
            # mindformers的约束
            if dp * tp % ep != 0:
                continue
            new_config = [dp, tp, pp, ep, initial_offset(pp, layers_num)]
            dry_run_config.append(new_config)

    logger.info(f"Dryrun config size: {len(dry_run_config)}")
    logger.info('dryrun configs format: dp_tp_pp_ep_offset')
    for config in dry_run_config:
        config_str = "_".join(str(x) for x in config)
        logger.info(config_str)
    return dry_run_config

