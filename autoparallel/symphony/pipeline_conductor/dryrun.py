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

import os
import shutil

import time

import yaml

import argparse

from pipeline_conductor import pp_util

from multiprocessing import Pool

# 一次起dryrun的进程数
dryrun_lim = 16
# 2为swap的dryrun,3为boss的dryrun,其余为原始dryrun
dryrun_mode = 1
# 开大内存：0；开小内存：1
ENABLE_LESS_MEM_VPP = '0'
pipeline_output_file = 'pipeline_output'
dryrun_yaml_dir = 'dryrun_yaml'


class DryRun:
    def __init__(self, yaml_file, mind_former_file, output_name):
        self.yaml_file = yaml_file
        self.mind_former_file = mind_former_file
        self.rank_gap = None
        pp_output_file = os.path.join(os.getcwd(), pipeline_output_file)
        if not os.path.exists(pp_output_file):
            os.mkdir(pp_output_file)
        self.log_file_name = os.path.join(pp_output_file, output_name)
        if os.path.exists(self.log_file_name):
            shutil.rmtree(self.log_file_name)
        os.mkdir(self.log_file_name)
        self.dryrun_lim = dryrun_lim
        self.dryrun_mode = dryrun_mode

    def dryrun(self, stage_num, rank_size):
        self.rank_gap = rank_size // stage_num
        if dryrun_mode == 3:
            self.set_environment_boss(rank_size)
        else:
            self.set_environment(rank_size)
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

    def set_environment(self, rank_size):
        os.environ['ENABLE_LESS_MEM_VPP'] = ENABLE_LESS_MEM_VPP
        os.environ['RANK_SIZE'] = str(rank_size)

        os.environ['MS_MEMORY_POOL_RECYCLE'] = '1'
        os.environ['GE_NOT_CUT'] = '1'
        os.environ['ENABLE_CELL_REUSE'] = '1'
        os.environ['MS_DEV_SIDE_EFFECT_LOAD_ELIM'] = '3'
        os.environ['ENABLE_LAZY_INLINE_NO_PIPELINE'] = '1'

        if self.dryrun_mode == 2:
            os.environ['MS_DEV_RUNTIME_CONF'] = 'inline:True,input_optimize:false,memory_statistics:True,compile_statistics:True,all_finite:False'
            # 关闭0，打开1
            os.environ['OPTIMIZER_OFFLOAD'] = '1'
        else:
            os.environ['MS_DEV_RUNTIME_CONF'] = 'memory_statistics:True,compile_statistics:True,all_finite:False'
            os.environ['OPTIMIZER_OFFLOAD'] = '0'

        os.environ['MS_DEV_DUMP_IR_PASSES'] = 'step_parallel,validate'
        os.environ['MS_ALLOC_CONF'] = 'enable_vmm:False,,older_pool:true'

        os.environ['MS_DATASET_SINK_QUEUE'] = '8'

        os.environ['HCCL_CONNECT_TIMEOUT'] = '3600'
        os.environ['HCCL_EXEC_TIMEOUT'] = '3600'
        os.environ['MS_RECEIVE_MSG_TIMEOUT'] = '3600'
        os.environ['MS_DISABLE_HEARTBEAT'] = '1'
        os.environ['MS_NODE_TIMEOUT'] = '3600'
        os.environ['MS_RETRY_INTERVAL_LOWER'] = '5'
        os.environ['MS_RETRY_INTERVAL_UPPER'] = '10'
        os.environ['MS_ASCEND_CHECK_OVERFLOW_MODE'] = 'INFNAN_MODE'

        os.environ['MS_MEMORY_STATISTIC'] = '1'
        os.environ['MS_SIMULATION_LEVEL'] = '1'
        # 1：错误日志；2：告警日志；3：重要日志；4：调试日志
        os.environ['GLOG_v'] = '2'

    def set_environment_boss(self, rank_size):
        os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
        os.environ['ENABLE_LESS_MEM_VPP'] = ENABLE_LESS_MEM_VPP
        os.environ['RANK_SIZE'] = str(rank_size)
        os.environ['HCCL_INTRA_PCIE_ENABLE'] = '0'
        os.environ['HCCL_INTRA_ROCE_ENABLE'] = '1'
        os.environ['ENABLE_LAZY_INLINE_NO_PIPELINE'] = '1'
        os.environ['MS_DEV_RUNTIME_CONF'] = "memory_statistics:True,inline:true"
        os.environ['MS_DEV_DUMP_IR_PASSES'] = "hwopt_d_after_stream_assign,valid"
        os.environ['MS_ALLOC_CONF'] = "memory_tracker:True"
        os.environ['MS_MEMORY_STATISTIC'] = '1'
        os.environ['MS_SIMULATION_LEVEL'] = '1'
        # # 1：错误日志；2：告警日志；3：重要日志；4：调试日志
        os.environ['GLOG_v'] = '2'

    def start_dryrun(self, recompute_config, offset, num_layers, num_vpp, num_stage, rank_size, dense_layers, micro):
        with open(self.yaml_file, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            data['recompute_config'] = recompute_config
            data['model']['model_config']['offset'] = offset
            if 'mtp_depth' in data['model']['model_config']:
                mtp_depth = data['model']['model_config']['mtp_depth']
                num_layers -= mtp_depth
            data['model']['model_config']['num_layers'] = num_layers
            if 'moe_config' in data:
                if 'first_k_dense_replace' in data['moe_config']:
                    data['moe_config']['first_k_dense_replace'] = dense_layers
            data['parallel_config']['pipeline_stage'] = num_stage
            data['parallel_config']['micro_batch_num'] = micro
            if 'pp_interleave_num' in data['model']['model_config']:
                data['model']['model_config']['pp_interleave_num'] = num_vpp
        timestamp = time.strftime("%Y%m%d%H%M%S")
        pipeline_output = os.path.join(os.getcwd(), pipeline_output_file)
        if not os.path.exists(pipeline_output):
            os.mkdir(pipeline_output)
        new_file = os.path.join(pipeline_output, dryrun_yaml_dir)
        if not os.path.exists(new_file):
            os.mkdir(new_file)
        name = os.path.join(new_file, f'{timestamp}.yaml')
        with open(name, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, default_flow_style=False, indent=4)
        self.yaml_file = name
        self.dryrun(num_stage, rank_size)

    def run_rank(self, stage):
        device_id = stage
        rank_id = stage * self.rank_gap
        cwd = os.getcwd()
        log_file = os.path.join(cwd, self.log_file_name, f'rank_{rank_id}.log')
        if self.dryrun_mode == 0:
            os.environ['DEVICE_ID'] = str(device_id)
        else:
            os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(device_id)
        os.environ['RANK_ID'] = str(rank_id)

        command = (f'python {self.mind_former_file} --register_path research/jiutian '
                   f'--config {self.yaml_file} &> {log_file}')
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
    dry_run.set_environment(rank_size)
    dry_run.run_rank(stage)


def all_rank_dryrun(yaml_file, mindformer_file, output_file):
    dry_run = DryRun(yaml_file, mindformer_file, output_file)
    rank_size, pipeline_stage = pp_util.get_ranks_stages(yaml_file)
    dry_run.dryrun(pipeline_stage, rank_size)
    print(dry_run.extract_memory_info_act(pipeline_stage))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Yaml config dryrun',
                                     description='Write config to the yaml file, and dryrun it', epilog='')
    parser.add_argument('--yaml', '-y', type=str, required=True, default=None,
                        help="Path of training config (.yaml)")
    parser.add_argument('--mindformers', '-m', type=str, required=True, default=None,
                        help="Absolute path of run_mindformers (.py)")
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
    args = parser.parse_args()

    yaml_file = args.yaml
    mindformer_file = args.mindformers
    output_file = args.output_file
    if args.recompute_layers and args.is_recompute is None:
        args.is_recompute = True
    if args.select_recompute_layers and args.is_select_recompute is None:
        args.is_select_recompute = True

    if args.offset is None and args.is_select_recompute is None and args.is_recompute is None:
        logger.info('Use old yaml config to dryrun')
    else:
        yaml_file = pp_util.build_new_config_yaml(args)

    all_rank_dryrun(yaml_file, mindformer_file, output_file)

    # one rank
    # one_rank_dryrun(0, yaml_file, mindformer_file, output_file)
