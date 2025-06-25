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
import csv
from typing import List
from pathlib import Path

from utils.profiling.profile_info import ProfileInfo
from pipeline_conductor.start_service import ExpertInput
from pipeline_conductor.dryrun import DryRun, dryrun_config_error

class ParallelConfig:
    def __init__(self, gbs, config):
        self.dp = config[0]
        self.tp = config[1]
        self.pp = config[2]
        self.vp = 1
        self.ep = config[3]
        self.micro = gbs / self.dp

class PipelineInputConfig:
    def __init__(self, profiling_info: ProfileInfo, config_path):
        self.profiling_info = profiling_info
        self.config_path = config_path


class ParallelInput:
    def __init__(self, args):
        self.candidate_configs: List[PipelineInputConfig] = []
        self.init_configs_info(args)

        self.is_lowmem = int(os.getenv('ENABLE_LESS_MEM_VPP', 0))
        self.solver_name = args.solver_name
        self.env_config_json = args.env_config_json
        self.register_path = args.register_path
        self.dryrun_lim = args.dryrun_lim
        self.dryrun = args.dryrun
        self.check = args.check
        if DryRun.config_file_type == 0:
            self.ms_adapter_file = args.mindformers_dir
        elif DryRun.config_file_type == 1:
            self.ms_adapter_file = args.mindspeed_dir
        else:
            raise TypeError(dryrun_config_error)

    @staticmethod
    def parse_results_by_csv(csv_file):
        result_dict = {}
        with open(csv_file, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                key = "_".join([row['dp'], row['tp'], row['pp'], row['ep']])
                value = [float(row['dmratio']), float(row['bfratio']), float(row['re_grow_ration']),
                         float(row['hratio']), float(row['moe_bw'])]
                result_dict[key] = value
        return result_dict

    def init_configs_info(self, args):
        if args.yaml_path and args.mindformers_dir:
            directory = Path(args.yaml_path)
            config_files = directory.glob('*.yaml')
            DryRun.config_file_type = 0
            ExpertInput.is_full_recomp = True
        elif args.shell_path and args.mindspeed_dir:
            directory = Path(args.shell_path)
            config_files = directory.glob('*.sh')
            DryRun.config_file_type = 1
            ExpertInput.is_full_recomp = False
        else:
            raise TypeError(dryrun_config_error)
        csv_result = {}
        # 若用户直接输入profiling解析信息---csv文件，则从文件中读入
        if args.parser_result is not None:
            csv_result = self.parse_results_by_csv(args.parser_result)

        for config_file in config_files:
            pattern = re.compile(r'DP(\d+)_TP(\d+)_PP(\d+)_EP(\d+)')
            match = pattern.search(config_file.name)
            if match:
                dp_num, tp_num, pp_num, ep_num = map(int, match.groups())
                config = [dp_num, tp_num, pp_num, ep_num]
                config_str = "_".join(map(str, config))
                if config_str in csv_result:
                    profile_list = csv_result[config_str]
                else:
                    profile_list = []
               # todo: profiling解析的输入有待确认,例如rank
                profile_info = ProfileInfo(args.profile_data_dir, profile_list)
                pipeline_input_config = PipelineInputConfig(profile_info, config_file)
                self.candidate_configs.append(pipeline_input_config)