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

import os
import subprocess
import re
import json
from multiprocessing import Pool
from utils.common import MODULE_PATTERN_REG, cal_model_layers_num, cal_world_size
from utils.logger import logger


def env_update(rank_size, env_variable_json):
    # 设置环境变量
    with open(env_variable_json, 'r') as f:
        env_vars = json.load(f)
    env_vars['RANK_SIZE'] = str(rank_size)
    os.environ.update(env_vars)

def create_target_dir(file, target_directory):
    output_dir = os.path.splitext(file)[0]
    target_dir = os.path.join(target_directory, f"dryrun_{output_dir}")
    return target_dir

def execute_command(para, yaml_file_path, target_dir, rank_id, ep, tp):
    command = ['python', os.path.join(para.MINDFORMERS_DIR, 'run_mindformer.py'), '--config', yaml_file_path]
    try:
        # 构建日志文件路径
        os.environ['RANK_ID'] = str(rank_id)
        os.environ['EXPERT_PARALLEL'] = str(ep)
        os.environ['MODEL_PARALLEL'] = str(tp)
        os.makedirs(target_dir, exist_ok=True)
        log_file_path = os.path.join(target_dir, 'dryrun_new.log')
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            logger.info(f"Command: {' '.join(command)} rank {rank_id} start run")
            subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
    except Exception as e:
        logger.error(f"The command execution failed.: {e}")

def calculate_rank_id(match, layer_num, rank_size):
    pp = int(match.group(3))
    x = pp - layer_num % pp
    rank_id = x * rank_size // pp
    if rank_id == rank_size:
        rank_id -= 1
    return rank_id

def read_dryrun_info(root_dir):
    logger.info(f"Reading dryrun info from {root_dir}")
    result_list = []
    # 预编译正则表达式，提高匹配性能
    pattern = re.compile(r'DP(\d+)_TP(\d+)_PP(\d+)_EP(\d+)')

    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue

        log_file_path = os.path.join(entry.path, 'dryrun_new.log')
        if not os.path.isfile(log_file_path):
            continue

        with open(log_file_path, 'r', encoding='utf-8') as log_file:
            for line in log_file:
                if "Used peak memory usage (without fragments):" in line:
                    peak_value = line.split(': ')[-1].strip()
                    peak_value = int(peak_value.rstrip('M'))
                    break
            else:
                # 如果没有找到 peak 值，跳过当前目录
                continue

        match = pattern.search(entry.name)
        if match:
            dp_num, tp_num, pp_num, ep_num = map(int, match.groups())
            result_list.append([dp_num, tp_num, pp_num, ep_num, peak_value])

    logger.info(f"read dryrun info done, result size {len(result_list)} format [dp, tp, pp, ep, peak_value]")
    logger.info('\n'.join(str(item) for item in result_list))
    return result_list

def launch_dryrun(mindformers_args, yaml_file_directory, dryrun_data_dir, para):
    logger.info('start dryrun')
    rank_size = cal_world_size(mindformers_args)
    env_update(rank_size, para.ENV_JSON)
    tasks = []
    layer_num = cal_model_layers_num(mindformers_args)

    # 遍历指定目录下的所有文件
    for root, _, files in os.walk(yaml_file_directory):
        for file in files:
            match = re.match(MODULE_PATTERN_REG, file)
            if not file.endswith('.yaml') or not match:
                continue
            rank_id = calculate_rank_id(match, layer_num, rank_size)
            ep = int(match.group(4))
            tp = int(match.group(2))
            target_dir = create_target_dir(file, dryrun_data_dir)
            yaml_file_path = os.path.join(root, file)
            tasks.append((para, yaml_file_path, target_dir, rank_id, ep, tp))

    with Pool(processes=para.PARALLEL_NUM) as pool:
        pool.starmap(execute_command, tasks)
    logger.info("all dryrun done")