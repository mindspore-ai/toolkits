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

import argparse
import csv
import time
from utils.input_config import InputConfig
from utils.input_param import InputParam
from utils.logger import logger

from ndsearch.memory_model import filter_oom_by_dryrun
from utils.profiling.profile_prepare import profile_prepare
from ndsearch.build_initial_spaces import build_initial_spaces
from ndsearch.expert_filter_configs import expert_filter_configs

__all__ = ['taylor_search_tool']


def taylor_search_tool(para):
    """
    A function for find out optimal ND parallel configuration.

    Args:
        param para: 用户输入自定义的参数

    Returns:
        parallel config fill back to yaml file: [dp, cp, tp, ...] and candidate csv file.
    """
    start_time = time.time()
    para.print_params()

    mindformers_args = InputConfig(para.YAML_PATH)

    initial_configs = build_initial_spaces(mindformers_args)
    logger.info(f"Initial Search space size: {len(initial_configs)}")

    expert_prune_search_space = expert_filter_configs(initial_configs, mindformers_args, para.GBS)
    logger.info(f"Expert Prune Search space size: {len(expert_prune_search_space)}")

    # [dp, tp, pp, ep, evaluate_peak_mem]
    dryrun_prune_space = filter_oom_by_dryrun(expert_prune_search_space, mindformers_args, para)
    logger.info(f"Dryrun Prune Search space size: {len(dryrun_prune_space)}, format: [dp, tp, pp, ep, evaluate_peak_mem]")
    print(*(config for config in dryrun_prune_space), sep='\n')
    write_result_to_csv(dryrun_prune_space)
    logger.info('%s', '\n'.join(str(item) for item in dryrun_prune_space))

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"program before profiling cost time: {elapsed_time} s")

    profile_prepare(dryrun_prune_space, para)

def write_result_to_csv(dryrun_prune_space):
    headers = ['dp', 'tp', 'pp', 'ep', 'dmratio', 'bfratio', 'hratio', 'moe_bw']
    # 写入 CSV 文件
    try:
        with open('final_result.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 写入表头
            writer.writerow(headers)
            # 写入数据
            writer.writerows(dryrun_prune_space[:-1])
        print("CSV file generate succ。")
    except Exception as e:
        print(f"write CSV file fail: {e}")


if __name__ == '__main__':
    logger.info('start to run parallel tool')
    parser = argparse.ArgumentParser(description='Run taylor_search_tool with user input parameters')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['YAML_PATH']}", type=str,
                        default='./config/pretrain_deepseek3_671b.yaml',
                        help='Path to the YAML file')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['MINDFORMERS_DIR']}", type=str,
                        default='./mindformers',
                        help='Directory of mindformers')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['DRYRUN_DATA_DIR']}", type=str, nargs='?', default='',
                        help='Directory of dryrun data')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['PROFILE_DATA_DIR']}", type=str,
                        default='./profile_data/',
                        help='Directory of profile data')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['PARALLEL_NUM']}", type=int, default=2,
                        help='Number of parallel processes')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['RANK_NUM']}", type=int, default=64,
                        help='Number of available device number')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['SOLVER_NAME']}", type=str, default='HIGHS',
                        help='Name of the solver')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['REGISTER_PATH']}", type=str, default='./register/',
                        help='Path of register')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['SMALL_DATASET']}", type=str,
                        default='./config/dataset/wiki4096.mindrecord',
                        help='Directory of small dataset')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['DATASET']}", type=str,
                        default='./config/dataset/wiki103-4k.mindrecord',
                        help='Directory of dataset')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['MAX_EXPERT_PARALLEL']}", type=int, default=64,
                        help='Max number of expert parallel')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['OUTPUT_PATH']}", type=str,
                        default='./output/nd_output/',
                        help='Directory of output info')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['ENV_JSON']}", type=str,
                        default='./config/env_config.json',
                        help='Environment variable config json file')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['GBS']}", type=int, default=1024,
                        help='Global batch size')

    args = parser.parse_args()
    para = InputParam(args)
    taylor_search_tool(para)
