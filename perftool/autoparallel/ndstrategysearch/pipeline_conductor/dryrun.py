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
import shutil
import json
import argparse
from utils.logger import logger
from pipeline_conductor import pp_util
from multiprocessing import Pool
from pipeline_conductor.pp_util import pipeline_output_file

dryrun_config_error = 'The config_file location and ms_adapter_file location are essential, please config!'


class DryRun:
    env_config_json = ''
    register_path = 'research/jiutian'
    dryrun_lim = 16
    config_file_type = 0
    is_write_to_file = True

    def __init__(self, config_file, ms_adapter_file, output_name):
        self.config_file = config_file
        self.ms_adapter_file = ms_adapter_file
        self.rank_gap = None
        pp_output_file = os.path.join(os.getcwd(), pipeline_output_file)
        if not os.path.exists(pp_output_file):
            os.mkdir(pp_output_file)
        self.log_file_name = os.path.join(pp_output_file, output_name)
        if os.path.exists(self.log_file_name):
            shutil.rmtree(self.log_file_name)
        os.mkdir(self.log_file_name)

    def dryrun(self, stage_num, rank_size):
        self.rank_gap = rank_size // stage_num
        self.set_env(rank_size, self.env_config_json)
        remainder = stage_num
        device_id = 0
        while remainder > 0:
            if remainder > self.dryrun_lim:
                dryrun_num = self.dryrun_lim
            else:
                dryrun_num = remainder
            remainder -= dryrun_num
            with Pool(processes=dryrun_num) as pool:
                pool.map(self.run_rank, range(device_id, device_id + dryrun_num))
            device_id += dryrun_num
        logger.info('pull dryrun of all stages!')

    def set_env(self, rank_size, env_config_json):
        with open(env_config_json, 'r') as f:
            env_vars = json.load(f)
        env_vars['RANK_SIZE'] = str(rank_size)
        os.environ.update(env_vars)

    def start_dryrun(self, recompute_config, offset, num_layers, num_vpp, num_stage, rank_size, dense_layers, micro):
        if self.config_file_type == 0:
            name = pp_util.bulid_yaml(self.config_file, recompute_config, offset,
                                      num_layers, num_vpp, num_stage, dense_layers, micro)
        elif self.config_file_type == 1:
            name = pp_util.bulid_shell(self.config_file, offset, num_layers, num_vpp, num_stage, dense_layers, micro)
        else:
            raise TypeError(dryrun_config_error)
        self.config_file = name
        self.dryrun(num_stage, rank_size)

    def run_rank(self, stage):
        device_id = stage
        rank_id = stage * self.rank_gap
        cwd = os.getcwd()
        log_file = os.path.join(cwd, self.log_file_name, f'rank_{rank_id}.log')
        if self.config_file_type == 0:
            os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(device_id)
            os.environ['RANK_ID'] = str(rank_id)
            command = (f'python {self.ms_adapter_file} --register_path {self.register_path} '
                       f'--config {self.config_file} &> {log_file}')
        elif self.config_file_type == 1:
            command = (f'export RANK_ID={rank_id}; '
                       f'bash {self.config_file} {device_id} {self.ms_adapter_file} {log_file}')
        else:
            raise TypeError(dryrun_config_error)
        logger.info(f"start training for rank_{rank_id}, device_{device_id}, waiting a moment...")
        pp_util.execute_command(command)

    def extract_memory_info(self, num_stage):
        cwd = os.getcwd()
        peak_mem = []
        for stage in range(num_stage):
            rank_id = self.rank_gap * stage
            log_file = os.path.join(cwd, self.log_file_name, f"rank_{rank_id}.log")
            peak_mem.append(pp_util.extract_peak_memory(log_file))
        return peak_mem

    def extract_memory_info_act(self, num_stage):
        cwd = os.getcwd()
        peak_mem = []
        for stage in range(num_stage):
            rank_id = self.rank_gap * stage
            log_file = os.path.join(
                cwd, self.log_file_name, f"rank_{rank_id}.log")
            peak_mem.append(pp_util.extract_actual_peak_memory(log_file))
        return peak_mem


def one_rank_dryrun(stage, yaml_file, mindformer_file, output_file):
    dry_run = DryRun(yaml_file, mindformer_file, output_file)
    rank_size, pipeline_stage = pp_util.get_ranks_stages(yaml_file)
    dry_run.rank_gap = rank_size // pipeline_stage
    dry_run.set_env(rank_size, dry_run.env_config_json)
    dry_run.run_rank(stage)


def all_rank_dryrun(config_file, ms_adapter_file, output_file):
    dry_run = DryRun(config_file, ms_adapter_file, output_file)
    if DryRun.config_file_type == 0:
        rank_size, pipeline_stage = pp_util.get_ranks_stages(config_file)
    elif DryRun.config_file_type == 1:
        rank_size, pipeline_stage = pp_util.get_shell_ranks_stages(config_file)
    else:
        raise TypeError(dryrun_config_error)
    dry_run.dryrun(pipeline_stage, rank_size)
    print(dry_run.extract_memory_info(pipeline_stage))
    print(dry_run.extract_memory_info_act(pipeline_stage))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Yaml config dryrun',
                                     description='Write config to the yaml file, and dryrun it', epilog='')
    parser.add_argument('--yaml', '-y', type=str, default=None,
                        help="Path of training config (.yaml)")
    parser.add_argument('--shell', '-sh', type=str, default=None,
                        help="Path of training config (.sh)")
    parser.add_argument('--mindformers', '-mf', type=str, default=None,
                        help="Absolute path of run_mindformers (.py)")
    parser.add_argument('--mindspeed', '-mp', type=str, default=None,
                        help="Absolute path of posttrain_gpt (.py)")
    parser.add_argument('--output_file', '-f', type=str, default='dryrun_output',
                        help="The location to place the output files")
    parser.add_argument('--offset', '-o', type=pp_util.str2list,
                        default=None, help="offset list")
    parser.add_argument('--is_recompute', '-ir', type=pp_util.str2bool,
                        default=None, help="Whether to open recompute")
    parser.add_argument('--recompute_layers', '-rl', type=pp_util.str2list,
                        default=None, help="recompute_layers list")
    parser.add_argument('--is_select_recompute', '-is', type=pp_util.str2bool,
                        default=None, help="Whether to open select_recompute")
    parser.add_argument('--select_recompute_layers', '-sl', type=pp_util.str2list,
                        default=None, help="select_recompute_layers list")
    parser.add_argument('--env_config_json', '-e', type=str, required=True,
                        default='./config/boss_env_config.json', help="Path of environment config (.json)")
    parser.add_argument('--register_path', '-rp', type=str, default='research/jiutian',
                        help="Path of register")
    parser.add_argument('--dryrun_lim', '-dl', type=pp_util.str2int, default=16,
                        help="The number of dryrun at once")
    args = parser.parse_args()

    if args.yaml and args.mindformers:
        config_file = args.yaml
        ms_adapter_file = args.mindformers
        DryRun.config_file_type = 0
    elif args.shell and args.mindspeed:
        config_file = args.shell
        ms_adapter_file = args.mindspeed
        DryRun.config_file_type = 1
    else:
        raise TypeError(dryrun_config_error)

    output_file = args.output_file
    DryRun.env_config_json = args.env_config_json
    DryRun.register_path = args.register_path
    DryRun.dryrun_lim = args.dryrun_lim
    if args.recompute_layers and args.is_recompute is None:
        args.is_recompute = True
    if args.select_recompute_layers and args.is_select_recompute is None:
        args.is_select_recompute = True

    if args.offset is None and args.is_select_recompute is None and args.is_recompute is None:
        logger.info('Use old yaml config to dryrun')
    elif DryRun.config_file_type == 0:
        config_file = pp_util.build_new_config_yaml(args)
    elif DryRun.config_file_type == 1:
        config_file = pp_util.build_new_config_shell(args)
    else:
        raise TypeError(dryrun_config_error)

    all_rank_dryrun(config_file, ms_adapter_file, output_file)

    # one rank
    # one_rank_dryrun(0, yaml_file, mindformer_file, output_file)
