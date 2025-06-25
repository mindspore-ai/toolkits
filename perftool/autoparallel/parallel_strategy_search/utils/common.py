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
import re

from pathlib import Path
import shutil
import yaml
import numpy as np

from pipeline_conductor.pp_util import configs_to_shell, parse_shell, parse_shell_config, flatten_to_str, \
    change_shell_config
from utils.logger import logger


def cal_model_layers_num(input_args):
    if input_args.model.model_config.mtp_depth is None:
        input_args.model.model_config.mtp_depth = 0
    return input_args.model.model_config.num_layers + input_args.model.model_config.mtp_depth

MODULE_PATTERN_YAML = 'DP{}_TP{}_PP{}_EP{}_pretrain.yaml'
MOE_PATTERN_SHELL = 'DP{}_TP{}_PP{}_EP{}_pretrain.sh'
LLAMA_PATTERN_SHELL = 'DP{}_TP{}_PP{}_pretrain.sh'
MODULE_PATTERN_REG_YAML = r'DP(\d+)_TP(\d+)_PP(\d+)_EP(\d+)_pretrain.yaml'
MOE_PATTERN_REG_SHELL = r'DP(\d+)_TP(\d+)_PP(\d+)_EP(\d+)_pretrain.sh'
LLAMA_PATTERN_REG_SHELL = r'DP(\d+)_TP(\d+)_PP(\d+)_pretrain.sh'

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

def offset_for_dualpipe(pipeline_stage, num_layers):
    pp_interleave_num = 2
    offset = np.zeros((pp_interleave_num, pipeline_stage), dtype=int).tolist()
    new_layers = num_layers - 2
    remainder = new_layers % pipeline_stage
    base = new_layers // pipeline_stage
    origin_base = new_layers // pipeline_stage

    for stage in range(pipeline_stage):
        # 获取当前stage的层数
        if remainder == 0:
            cur_layer = base
        else:
            cur_layer = base + 1
            remainder -= 1
        # 给vpp分配层
        vpp1_layer = cur_layer // pp_interleave_num
        if pipeline_stage - stage - 1 == 0:
            offset[0][pipeline_stage - stage - 1] = vpp1_layer + 2 - origin_base
        else:
            offset[0][pipeline_stage - stage - 1] = vpp1_layer - origin_base
        vpp2_layer = cur_layer - vpp1_layer
        offset[1][stage] = vpp2_layer - origin_base
    return offset

def cal_world_size(input_args):
    world_size = input_args.dp * \
                     input_args.tp * \
                     input_args.pp
                    # input_args.parallel_config.context_parallel
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
        # 重计算设置为true
        yaml_data['recompute_config']['recompute'] = True

        # todo: 适配tnd
        yaml_data['train_dataset']['data_loader']['dataset_dir'] = para.DATASET


    # 将修改后的数据写回YAML文件
    with open(destination_file, 'w', encoding='utf-8') as file:
        yaml.dump(yaml_data, file, default_flow_style=False, allow_unicode=True)

    logger.info(f"The dryrun YAML file copied and modified, new file is: {destination_file}")

def generate_dryrun_shell(destination_file, config, para):
    """

        :param para: 用户输入参数
        :param destination_file: 修改并行参数的配置shell文件
        :param config: [dp, tp, pp, ep, offset] or [dp, tp, pp]
        :return:
    """
    configs, unparses = parse_shell(para.SHELL_PATH)
    configs['DP'] = config[0]
    configs['TP'] = config[1]
    configs['PP'] = config[2]
    configs_to_shell(destination_file, configs, unparses)
    logger.info(f'The dryrun SHELL file copied and modified, new file is {destination_file}')

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

def generate_profile_shell(destination_file, config, para):
    """
    :param para: 用户输入参数

    :param destination_file:
    :param config: [dp, tp, pp, ep, offset, num_layers]
    :return:
    """
    configs, unparses = parse_shell(para.SHELL_PATH)
    configs['DP'] = config[0]
    configs['TP'] = config[1]
    configs['PP'] = config[2]
    # 生成要分析的进程编号列表
    step = para.RANK_NUM // config[2]
    profile_ranks = ' '.join(map(str, range(0, para.RANK_NUM, step)))
    output_dir = os.path.join(para.OUTPUT_PATH, "profile_result", f"DP{config[0]}_TP{config[1]}_PP{config[2]}_profile"
)
    # 使用f-string构建参数字符串，提高可读性
    profile_args = (
        "--profile "
        "--profile-step-start 5 "
        "--profile-step-end 7 "
        "--profile-level level1 "
        "--profile-with-stack "
        "--profile-with-cpu "
        f"--profile-ranks {profile_ranks} "  # 确保参数之间有空格
        f"--profile-save-path {output_dir}"
    )

    # 存入配置字典
    configs['PROFILE_ARGS'] = profile_args
    configs_to_shell(destination_file, configs, unparses)
    insert_profile_args_final(destination_file)
    logger.info(f'The profile SHELL file copied and modified, new file is {destination_file}')
    return output_dir

def insert_profile_args_final(shell_file_path, profile_args="$PROFILE_ARGS \\"):

    with open(shell_file_path, 'r') as f:
        lines = f.readlines()

    # 标记是否在 torchrun 命令块内
    in_torchrun_block = False
    modified_lines = []

    for line in lines:
        stripped = line.strip()

        # 检测命令块开始
        if stripped.startswith("msrun") and "pretrain_gpt.py" in stripped:
            in_torchrun_block = True
            modified_lines.append(line)
            modified_lines.append(f"    {profile_args}\n")
            continue

        # 检测命令块结束（最后一个参数行，以 \ 结尾）
        if in_torchrun_block and stripped.endswith("\\"):
            modified_lines.append(line)
            continue

        # 命令块结束后的行
        if in_torchrun_block:
            in_torchrun_block = False

        # 普通行
        modified_lines.append(line)

    # 写入修改后的内容
    with open(shell_file_path, 'w') as f:
        f.writelines(modified_lines)

    logger.info(f"insert PROFILE_ARGS to {shell_file_path}")


# 定义一个映射，将yaml_task和对应的生成函数关联起来
TASK_FUNCTION_MAP = {
    "dryrun_yaml": generate_dryrun_yaml,
    "dryrun_shell": generate_dryrun_shell,
    "profile_yaml": generate_profile_yaml,
    "profile_shell": generate_profile_shell,
}

def generate_files(candidate_configs, des_file_directory, file_task, para, input_args):
    # 如果目录不存在，则创建它
    if not os.path.exists(des_file_directory):
        os.makedirs(des_file_directory)

    # 检查file_task是否有效
    if file_task not in TASK_FUNCTION_MAP:
        logger.error(f"Invalid file_task value: {file_task}. Please use one of {list(TASK_FUNCTION_MAP.keys())}.")
        return

    generate_function = TASK_FUNCTION_MAP[file_task]
    if para.SHELL_PATH:
        if input_args.expert_num == 0:
            pattern = LLAMA_PATTERN_SHELL
        else:
            pattern = MOE_PATTERN_SHELL
    else:
        pattern = MODULE_PATTERN_YAML

    file_path_list = []
    output_dir_list = []
    for config in candidate_configs:
        # 生成输出文件路径,包括文件名
        destination_file = (des_file_directory + pattern).format(*config)
        file_path_list.append(destination_file)
        output_dir = generate_function(destination_file, config, para)
        output_dir_list.append(output_dir)
    return file_path_list, output_dir_list


def is_dualpipe_open(input_args):
    if input_args.mf_args is None:
        return False
    parallel_cfg = input_args.mf_args.parallel
    use_zero_bubble_v = (
            hasattr(parallel_cfg, 'pipeline_config') and
            hasattr(parallel_cfg.pipeline_config, 'pipeline_scheduler') and
            parallel_cfg.pipeline_config.pipeline_scheduler == 'zero_bubble_v'
    )
    return use_zero_bubble_v

# def get_args_from_file(para):
#     # if para.YAML_PATH:
#     #     input_args = InputConfig(para.YAML_PATH)
#     # elif para.SHELL_PATH:
#     #     # todo: 填入shell解析
#     #     input_args, unparses = parse_shell(para.SHELL_PATH)
#     para_nd = ParaForNd(para)
#     return para_nd