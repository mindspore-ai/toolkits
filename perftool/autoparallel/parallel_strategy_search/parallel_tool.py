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
from utils.common import check_dryrun_parallel_number
from pipeline_conductor import pp_util
from pipeline_conductor.pipeline_parallel import pipeline_proc
from utils.input_param import InputParam
from utils.logger import logger

from ndsearch.memory_model import filter_oom_by_dryrun
from utils.ppc_input import ParallelInput
from utils.profiling.profile_launch import ProfileLaunch
from utils.profiling.profile_parser import ProfileParser
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
    logger.info('%s', '\n'.join(str(item) for item in dryrun_prune_space))

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"program before profiling cost time: {elapsed_time} s")

    # profile_configs: list[dp, tp, pp, 0, 0, layers_num, profile_result_dir]
    # profile_file_dir: profile shell file dir
    profile_configs, profile_file_dir = profile_prepare(dryrun_prune_space, para, input_args)
    if para.ALG_PHASE != 1:
        logger.info(f"ALG_PHASE: {para.ALG_PHASE}, no need to profile and solve pipeline")
        return
    # 自动执行profile
    profile_launch = ProfileLaunch(profile_configs, para)
    profile_launch.profile_launch(profile_file_dir)
    # 自动profile解析
    profile_parser = ProfileParser(input_args.mbn, input_args.num_layers)
    # candidate_configs: List[PipelineInputConfig]
    candidate_configs = profile_parser.parse_batch_profile_result(profile_configs, para)

    # 流水线求解 todo: 想办法把candidate_configs传进去，不用csv读取
    pipeline_input = ParallelInput(para.args, profile_file_dir)
    pipeline_proc(pipeline_input)

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
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['MINDSPEED_PATH']}", type=str, nargs='?', default='/home/test/pretrain_gpt.py',
                        help='Path to the MindSpeed file')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['DRYRUN_DATA_DIR']}", type=str, nargs='?', default='',
                        help='Directory of dryrun data')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['PROFILE_DATA_DIR']}", type=str,
                        default='./profile_data/',
                        help='Directory of profile data')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['PARALLEL_NUM']}", type=int, default=2,
                        help='Number of parallel dryrun processes')
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
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['PARSER_RESULT']}", type=str, default=None,
                        help='Profiling parser result file')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['DRYRUN']}", type=pp_util.str2bool, default=True,
                        help='Is auto dryrun')
    parser.add_argument(f"--{InputParam.PARAM_MAPPING['CHECK']}", type=pp_util.str2bool, default=True,
                        help="Is double check")

    args = parser.parse_args()
    check_dryrun_parallel_number(args.parallel_num)
    para = InputParam(args)
    taylor_search_tool(para)
