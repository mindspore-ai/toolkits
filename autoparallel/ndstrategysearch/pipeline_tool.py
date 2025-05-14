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

from utils.logger import logger
from utils.ppc_input import ParallelInput
from utils.logger import logger

from pipeline_conductor.pipeline_parallel import pipeline_proc


if __name__ == '__main__':
    logger.info('start to run pipeline tool')
    # 用户输入profiling结果，候选配置等信息，流水线工具给出配置cost排序
    parser = argparse.ArgumentParser(description='Run taylor pipeline_search_tool with user input parameters')
    parser.add_argument('--yaml_path', type=str, default='./output/dryrun_yaml/',
                        help='Path to the YAML file directory')
    parser.add_argument('--mindformers_dir', type=str,
                        default='./mindformers/run_mindformer.py',
                        help='Directory of mindformers')
    parser.add_argument('--profile_data_dir', type=str, default='./profile_data/',
                        help='Directory of profile data')
    parser.add_argument('--solver_name', type=str, default='HIGHS',
                        help='Name of the solver')
    parser.add_argument('--parser_result', type=str, default=None,
                        help='Profiling parser result file')
    parser.add_argument('--gbs', type=int, default=1024,
                        help='Global batch size')
    parser.add_argument('--nd_path', type=str, default='./config/nd_result.csv',
                        help='Path to nd result file')

    args = parser.parse_args()
    pipeline_input = ParallelInput(args)
    pipeline_proc(pipeline_input)
