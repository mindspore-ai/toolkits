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

from ndsearch.para_for_nd_search import ParaForNd
from utils.input_param import InputParam
from utils.logger import logger

from ndsearch.memory_model import filter_oom_by_dryrun
from utils.profiling.profile_launch import ProfileLaunch
from utils.profiling.profile_parser import ProfileExe
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

    input_args = ParaForNd(para)

    initial_configs = build_initial_spaces(input_args)
    logger.info(f"Initial Search space size: {len(initial_configs)}")

    expert_prune_search_space = expert_filter_configs(initial_configs, input_args, para.GBS)
    logger.info(f"Expert Prune Search space size: {len(expert_prune_search_space)}")

    # [dp, tp, pp, ep, evaluate_peak_mem]
    dryrun_prune_space = filter_oom_by_dryrun(expert_prune_search_space, input_args, para)
    logger.info(f"Dryrun Prune Search space size: {len(dryrun_prune_space)}, format: [dp, tp, pp, ep, evaluate_peak_mem]")
    print(*(config for config in dryrun_prune_space), sep='\n')
    # write_result_to_csv(dryrun_prune_space)
    logger.info('%s', '\n'.join(str(item) for item in dryrun_prune_space))

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"program before profiling cost time: {elapsed_time} s")

    profile_configs, profile_file_dir = profile_prepare(dryrun_prune_space, para, input_args)

    # todo:自动执行profile
    profile_launch = ProfileLaunch(profile_configs, para)
    profile_launch.profile_launch(profile_file_dir)
    # 自动profile解析
    profile_exe = ProfileExe()
    # profile_exe.config_anal(profile_configs, ranks)


# 把profile结果写入csv
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
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['YAML_PATH']}", type=str, nargs='?', default='',
                        help='Path to the YAML file')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['SHELL_PATH']}", type=str, nargs='?', default='./config/pretrain_llama2_7b_ptd.sh',
                        help='Path to the SHELL type config file')
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
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['RANK_NUM']}", type=int, default=8,
                        help='Number of available device number')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['SOLVER_NAME']}", type=str, default='HIGHS',
                        help='Name of the solver')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['REGISTER_PATH']}", type=str, default='./register/',
                        help='Path of register')
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
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['SELECT_RECOMPUTE']}", type=bool, default=True,
                        help='Whether search select recompute')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['ALG_PHASE']}", type=int, default=1,
                        help='Phase of parallel strategy search algorithm')

    args = parser.parse_args()
    para = InputParam(args)
    taylor_search_tool(para)
