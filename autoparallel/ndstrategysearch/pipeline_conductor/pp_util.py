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

from utils.logger import logger
import numpy as np
import time
import re
import sys

import subprocess
import argparse

import chardet

import highspy

import yaml

import ast
import os


def update_yaml_value(yaml_file, key, value):
    with open(yaml_file, 'r', encoding='utf-8') as file:
        yaml_data = yaml.safe_load(file)
        if key in yaml_data:
            print(f"find the {key}, update the context")
        else:
            print(f"can not find {key}")
        yaml_data[key] = value
    with open(yaml_file, 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)


def highs_solve_mps(mps_file, solution_file, origin_model, time_limit):
    highs_solver = highspy.Highs()
    highs_solver.readModel(mps_file)
    if time_limit != sys.maxsize:
        highs_solver.setOptionValue('time_limit', time_limit)
    # highs_solver.setOptionValue('threads', 8)

    highs_solver.run()

    info = highs_solver.getInfo()
    origin_model.solution_status = highs_solver.solutionStatusToString(info.primal_solution_status)
    logger.info(f'Solution status = {origin_model.solution_status}')
    origin_model.model_status = highs_solver.modelStatusToString(highs_solver.getModelStatus())
    logger.info(f'Model status = {origin_model.model_status}')
    origin_model.run_time = highs_solver.getRunTime()
    logger.info(f'Run time = {origin_model.run_time}')
    origin_model.gap = info.mip_gap
    logger.info(f'Gap = {origin_model.gap * 100:.2f}%')
    logger.info(f'Optimal objective = {info.objective_function_value}')
    origin_model.min_time = info.objective_function_value
    if origin_model.model_status == 'Time limit reached' and origin_model.gap > 0:
        logger.info(f'Dual bound = {info.mip_dual_bound}')
    if origin_model.solution_status == 'None':
        if origin_model.model_status == 'Infeasible':
            logger.error(f'{mps_file} is Infeasible, Please check the memory limit!')
        elif origin_model.model_status == 'Time limit reached':
            logger.error(f'{mps_file} is not finished!, Please check the time limit!')
        else:
            logger.error(f'{mps_file} is no solution, model_status = {origin_model.model_status}!')
    highs_solver.writeSolution(solution_file, 0)
    logger.info(f'Writing the solution to {solution_file}')


def qiuqi_solver_mps(solver_file, model_file, solution_file_name, origin_model):
    command01 = f'chmod +x {solver_file} && {solver_file} -Help'
    execute_command(command01)
    command2 = f'{solver_file} -SolutionFile={solution_file_name} {model_file}'
    subprocess.run(command2, shell=True)
    logger.info(f'Writing the solution to {solution_file_name}')
    with open(solution_file_name, 'r', encoding='utf-8') as file:
        content = file.read()
        result_content = re.search(r'# Objective: (\d+\.\d+)', content)
        objective_function_value = result_content.group(1)
    logger.info(f'Optimal objective = {objective_function_value}')
    origin_model.min_time = objective_function_value


def execute_command(command):
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def write_config_to_yaml(recompute_config, offset, yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        data['recompute_config'] = recompute_config
        data['model']['model_config']['offset'] = offset
    with open(yaml_file, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, default_flow_style=False, indent=4)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(v):
    if isinstance(v, list):
        return v
    if isinstance(ast.literal_eval(v), list):
        return ast.literal_eval(v)
    else:
        raise argparse.ArgumentTypeError('List value expected.')


def build_new_config_yaml(args):
    with open(args.yaml, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        if args.offset:
            data['model']['model_config']['offset'] = args.offset
            logger.info(f'change offset to {args.offset}')
        recompute_config = build_recompute_config(args.is_select_recompute, args.is_recompute,
                                                  args.select_recompute_layers, args.recompute_layers)
        if args.is_select_recompute is None:
            logger.info(f'is_select_recompute is None')
        else:
            data['recompute_config']['select_recompute'] = recompute_config['select_recompute']
            data['recompute_config']['select_comm_recompute'] = recompute_config['select_comm_recompute']
            logger.info(f'change is_select_recompute to {args.is_select_recompute}')
            logger.info(f'change select_recompute_layers to {args.select_recompute_layers}')
        if args.is_recompute is None:
            logger.info(f'is_recompute is None')
        else:
            data['recompute_config']['recompute'] = recompute_config['recompute']
            logger.info(f'change is_recompute to {args.is_recompute}')
            logger.info(f'change recompute_layers to {args.recompute_layers}')
    file_name, file_ext = os.path.splitext(os.path.basename(args.yaml))
    yaml_path = os.path.dirname(os.path.abspath(args.yaml))
    timestamp = time.strftime("%Y%m%d%H%M%S")
    new_yaml = os.path.join(yaml_path, f'{file_name}_{timestamp}.yaml')
    with open(new_yaml, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, default_flow_style=False, indent=4)
    logger.info(f'build new yaml {new_yaml}')
    return new_yaml


def build_recompute_config(is_select_recompute, is_recompute, select_recompute_layers, recompute_layers):
    recompute_config = {
        'parallel_optimizer_comm_recompute': False,
        'mp_comm_recompute': True,
        'recompute_slice_activation': True
    }

    if is_select_recompute:
        # 选择对应层的算子进行重计算
        recompute_config['select_recompute'] = {}
        recompute_config['select_recompute'][r'feed_forward\.null'] = select_recompute_layers
        recompute_config['select_recompute'][r'feed_forward\.w1\.activation\.silu'] = select_recompute_layers
        recompute_config['select_recompute'][r'feed_forward\.w1\.reshape'] = select_recompute_layers
        recompute_config['select_recompute'][r'feed_forward\.w2\.reshape'] = select_recompute_layers
        recompute_config['select_recompute'][r'add'] = select_recompute_layers
        recompute_config['select_recompute'][r'cast_up'] = select_recompute_layers
        # 选择对应层的算子进行通信重计算
        recompute_config['select_comm_recompute'] = {}
        recompute_config['select_comm_recompute'][r'.*\.norm'] = select_recompute_layers
        recompute_config['select_comm_recompute'][r'attention\.wq\.reshape'] = select_recompute_layers
        recompute_config['select_comm_recompute'][r'attention\.wk\.reshape'] = select_recompute_layers
        recompute_config['select_comm_recompute'][r'feed_forward\.w1\.reshape'] = select_recompute_layers
        recompute_config['select_comm_recompute'][r'feed_forward\.w3\.reshape'] = select_recompute_layers
    elif is_select_recompute is False:
        recompute_config['select_recompute'] = False
        recompute_config['select_comm_recompute'] = False

    if is_recompute:
        recompute_config['recompute'] = recompute_layers
    elif is_recompute is False:
        recompute_config['recompute'] = False

    return recompute_config


def construct_distribution(num_micro, num_stage):
    parts, remainder = num_micro // num_stage, num_micro % num_stage
    distribution = []
    for part in range(parts):
        if part == 0:
            distribution.append(num_stage + remainder)
        else:
            distribution.append(num_stage)
    return distribution


def sort_micro(parts, num_vpp, num_stage, distribution, low_mem, seq_split, is_f_then_b: bool = False):
    forward = []
    backward = []
    final_orders = []
    for part in range(parts):
        for vpp in range(num_vpp):
            for micro_id in range(distribution[part]):
                for split in range(seq_split):
                    forward.append((part, vpp, 'f', micro_id, split))
        for vpp in range(num_vpp - 1, -1, -1):
            for micro_id in range(distribution[part]):
                for split in range(seq_split - 1, -1, -1):
                    backward.append((part, vpp, 'b', micro_id, split))
    # f-then-b的调度规则，待启用
    if is_f_then_b:
        for stage in range(num_stage):
            stage_order = []
            for part in range(parts):
                for micro_id in range(distribution[part]):
                    stage_order.append((part, 0, 'f', micro_id, 0))
                for micro_id in range(distribution[part]):
                    stage_order.append((part, 0, 'b', micro_id, 0))
            final_orders.append(stage_order)
        return final_orders

    for stage in range(num_stage):
        if low_mem:
            warmup = min(((num_vpp - 1) * distribution[0] + (num_stage - stage - 1)) * seq_split, len(forward))
        else:
            warmup = min(((num_vpp - 1) * distribution[0] + (num_stage - stage - 1) * 2) * seq_split, len(forward))
        # 最后一个stage，第一个micro前向做完之后才能做后向
        if stage == num_stage - 1:
            warmup = warmup + seq_split - 1
        stage_order = []
        stage_order += forward[: warmup]
        for i in range(warmup, len(forward)):
            stage_order.append(forward[i])
            stage_order.append(backward[i - warmup])
        stage_order += backward[len(forward) - warmup:]
        final_orders.append(stage_order)
    return final_orders


def get_init_input_peak_mem():
    return []


def get_state_input_peak_mem():
    return []


def get_ranks_stages(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        pipeline_stage = data['parallel_config']['pipeline_stage']
        model_parallel = data['parallel_config']['model_parallel']
        data_parallel = data['parallel_config']['data_parallel']
    rank_size = pipeline_stage * model_parallel * data_parallel
    return rank_size, pipeline_stage


def get_layers_distribution(offset, num_layers, num_stages, num_vpp):
    layers_stage_vpp = [[0 for _ in range(num_vpp)] for _ in range(num_stages)]
    for stage in range(num_stages):
        for vpp in range(num_vpp):
            layers_stage_vpp[stage][vpp] = offset[vpp][stage] + num_layers // (num_stages * num_vpp)
    return layers_stage_vpp


def find_most_times_stage(layer_num_of_stage):
    # key 为各个stage的layer数，value 为layer数对应的stage编号List
    frequency_dict = {}
    layer_num = layer_num_of_stage[0]
    max_time = 0
    for i in range(len(layer_num_of_stage)):
        if layer_num_of_stage[i] not in frequency_dict:
            frequency_dict[layer_num_of_stage[i]] = []
        frequency_dict[layer_num_of_stage[i]].append(i)
        if len(frequency_dict[layer_num_of_stage[i]]) > max_time:
            max_time = len((frequency_dict[layer_num_of_stage[i]]))
            layer_num = layer_num_of_stage[i]
    return frequency_dict[layer_num]


def extract_peak_memory(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    with open(file_path, 'r', encoding=encoding) as file:
        content = file.read()
        result_content = re.search(r'Used peak memory usage \(without fragments\):\s*(\d+)M', content)
        if result_content:
            return int(result_content.group(1))
        else:
            raise ValueError(f"There is not match content in {file_path}")


def extract_actual_peak_memory(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    with open(file_path, 'r', encoding=encoding) as file:
        content = file.read()
        result_content = re.search(
            r'Actual peak memory usage \(with fragments\):\s*(\d+)M', content)
        if result_content:
            return int(result_content.group(1))
        else:
            raise ValueError(f"There is not match content in {file_path}")


def get_peak_batch(micro_batch_num, num_stage, num_vpp, low_mem, offset, num_layers):
    distribution = construct_distribution(micro_batch_num, num_stage)
    final_orders = sort_micro(micro_batch_num // num_stage, num_vpp, num_stage, distribution, low_mem, 1)
    layers_stage_vpp = get_layers_distribution(offset, num_layers, num_stage, num_vpp)
    # 求峰值内存时的激活份数
    peaks = [0] * num_stage
    for stage in range(num_stage):
        cur_mem = 0
        for micro in final_orders[stage]:
            if micro[2] == 'f':
                cur_mem += layers_stage_vpp[stage][micro[1]]
            else:
                cur_mem -= layers_stage_vpp[stage][micro[1]]
            peaks[stage] = max(peaks[stage], cur_mem)
    return peaks


def build_coe_array(peak_num_act, peak_num_select_recom, peak_num_recompute, x, stage_range):
    num_stage = len(peak_num_act)
    layer_dis = [sum(x[vpp][stage] for vpp in range(1)) for stage in range(num_stage)]
    coe_a = np.empty((stage_range[-1] - stage_range[0], 5), float)
    for stage in range(stage_range[0], stage_range[-1]):
        coe_a[stage - stage_range[0]][0] = 1
        coe_a[stage - stage_range[0]][1] = peak_num_act[stage]
        coe_a[stage - stage_range[0]][2] = layer_dis[stage]
        coe_a[stage - stage_range[0]][3] = peak_num_recompute[stage]
        coe_a[stage - stage_range[0]][4] = peak_num_select_recom[stage]
    return coe_a

