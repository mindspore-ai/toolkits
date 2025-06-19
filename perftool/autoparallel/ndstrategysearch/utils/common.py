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

from pathlib import Path
import shutil
import yaml
import numpy as np

from utils.logger import logger


def cal_model_layers_num(mindformers_args):
    if mindformers_args.model.model_config.mtp_depth is None:
        mindformers_args.model.model_config.mtp_depth = 0
    return mindformers_args.model.model_config.num_layers + mindformers_args.model.model_config.mtp_depth

MODULE_PATTERN = 'DP{}_TP{}_PP{}_EP{}_pretrain.yaml'
MODULE_PATTERN_REG = r'DP(\d+)_TP(\d+)_PP(\d+)_EP(\d+)_pretrain.yaml'

def initial_offset(pipeline_stage, num_layers):
    if pipeline_stage == 1 or num_layers % pipeline_stage == 0:
        offset = 0
        return offset

    pp_interleave_num = 1
    offset = np.zeros((pp_interleave_num, pipeline_stage), dtype=int).tolist()
    remainder = num_layers % (pp_interleave_num * pipeline_stage)
    for vpp in range(pp_interleave_num):
        offset[0][0] += 1
        remainder-=1
        for stage in range(pipeline_stage):
            if remainder == 0:
                break
            offset[vpp][pipeline_stage - stage - 2] += 1
            remainder -= 1
    return offset[0]

def cal_world_size(mindformers_args):
    world_size = mindformers_args.parallel_config.data_parallel * \
                     mindformers_args.parallel_config.model_parallel * \
                     mindformers_args.parallel_config.pipeline_stage
                    # mindformers_args.parallel_config.context_parallel
    return world_size

def generate_dryrun_yaml(destination_file, config, para):
    """

    :param para: 用户输入参数
    :param destination_file: 修改并行参数的配置yaml文件
    :param config: [dp, tp, pp, ep]
    :return:
    """
    # 复制YAML文件
    shutil.copy2(para.YAML_PATH, destination_file)

    # 读取复制后的YAML文件, 修改config_para字段的值
    with open(destination_file, 'r', encoding='utf-8') as file:
        yaml_data = yaml.safe_load(file)
        yaml_data['parallel_config']['data_parallel'] = config[0]
        yaml_data['parallel_config']['model_parallel'] = config[1]
        yaml_data['parallel_config']['pipeline_stage'] = config[2]
        yaml_data['parallel_config']['expert_parallel'] = config[3]
        yaml_data['parallel_config']['micro_batch_num'] = para.GBS // config[0]
        yaml_data['model']['model_config']['offset'] = config[4]
        if yaml_data['parallel'].get('dataset_strategy') is not None:
            strategy_size = len(yaml_data['parallel']['dataset_strategy'])
            yaml_data['parallel']['dataset_strategy'] = [[config[0], 1] for _ in range(strategy_size)]

        # dp > 4 则需要使用大数据集
        if config[0] > 4:
            yaml_data['train_dataset']['data_loader']['dataset_dir'] = para.DATASET
        else:
            yaml_data['train_dataset']['data_loader']['dataset_dir'] = para.SMALL_DATASET

    # 将修改后的数据写回YAML文件
    with open(destination_file, 'w', encoding='utf-8') as file:
        yaml.dump(yaml_data, file, default_flow_style=False, allow_unicode=True)

    logger.info(f"The dryrun YAML file copied and modified, new file is: {destination_file}")

def generate_profile_yaml(destination_file, config, para):
    """
    :param para: 用户输入参数

    :param destination_file:
    :param config: [dp, tp, pp, ep, offset, num_layers]
    :return:
    """
    # 复制YAML文件
    shutil.copy2(para.YAML_PATH, destination_file)

    # 读取复制后的YAML文件, 修改config_para字段的值
    with open(destination_file, 'r', encoding='utf-8') as file:
        yaml_data = yaml.safe_load(file)
        yaml_data['parallel_config']['data_parallel'] = config[0]
        yaml_data['parallel_config']['model_parallel'] = config[1]
        yaml_data['parallel_config']['pipeline_stage'] = config[2]
        yaml_data['parallel_config']['expert_parallel'] = config[3]
        yaml_data['parallel_config']['micro_batch_num'] = para.GBS // config[0]
        yaml_data['model']['model_config']['offset'] = config[4]
        yaml_data['recompute_config']['recompute'] = config[5]
        yaml_data['model']['model_config']['num_layers'] = config[6]

        yaml_data['profile'] = True
        yaml_data['profile_output'] = os.path.join(Path(destination_file).parent, "output")
        yaml_data['init_start_profile'] = True
        yaml_data['profile_communication'] = True
        yaml_data['profile_memory'] = True
        yaml_data['op_time'] = True
        yaml_data['profile_level'] = 1
        yaml_data['profile_start_step'] = 4
        yaml_data['profile_stop_step'] = 6

    # 将修改后的数据写回YAML文件
    with open(destination_file, 'w', encoding='utf-8') as file:
        yaml.dump(yaml_data, file, default_flow_style=False, allow_unicode=True)

    logger.info(f"The profile YAML file copied and modified, config_para is: {config}")

# 定义一个映射，将yaml_task和对应的生成函数关联起来
TASK_FUNCTION_MAP = {
    "dryrun": generate_dryrun_yaml,
    "profile": generate_profile_yaml
}

def generate_files(candidate_configs, des_file_directory, yaml_task, para):
    # 如果目录不存在，则创建它
    if not os.path.exists(des_file_directory):
        os.makedirs(des_file_directory)

    # 检查yaml_task是否有效
    if yaml_task not in TASK_FUNCTION_MAP:
        logger.error(f"Invalid yaml_task value: {yaml_task}. Please use one of {list(TASK_FUNCTION_MAP.keys())}.")
        return

    generate_function = TASK_FUNCTION_MAP[yaml_task]

    yaml_path_list = []
    for config in candidate_configs:
        # 生成输出文件路径,包括文件名
        destination_file = (des_file_directory + MODULE_PATTERN).format(*config)
        yaml_path_list.append(destination_file)
        generate_function(destination_file, config, para)
    return yaml_path_list