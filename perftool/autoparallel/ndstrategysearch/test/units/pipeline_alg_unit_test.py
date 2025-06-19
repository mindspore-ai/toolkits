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

import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())
import unittest

from pipeline_conductor import dryrun, pipeline_parallel
from utils.logger import logger
from pipeline_conductor.start_service import InitConfig, ExpertInput

#流水线负载均衡算法单元测试用例
class PipeTestCase(unittest.TestCase):

    def get_yaml_path(self, filename):
        yaml_file_path = Path(__file__).resolve().parents[2] / 'config' / filename
        return yaml_file_path

    def test_micro64_dp16_tp4_pp16_vp1_ep8(self):
        yaml_file = self.get_yaml_path('test_62_16_1_64_1_0.yaml')
        mind_former_file = ''
        expert_input = ExpertInput(yaml_file, mind_former_file)
        expert_input.solver_name = 'HIGHS'
        expert_input.is_dryrun = False
        expert_input.backward_ratio = 1.839
        expert_input.layer_ratio = 0.397
        expert_input.head_loss = 1.5
        expert_input.recompute_ratio = 1
        model_input = InitConfig(expert_input)
        model_input.memory.select_mem0 = 400.0
        model_input.memory.select_mem12 = 400.0
        model_input.memory.select_mem = 765.0
        model_input.memory.re_comp_mem0 = 30.0
        model_input.memory.re_comp_mem12 = 30.0
        model_input.memory.re_comp_mem = 30.0
        model_input.memory.act_mem0 = 479.0
        model_input.memory.act_mem12 = 479.0
        model_input.memory.act_mem = 765.0
        model_input.memory.layer_mem012 = 866.0
        model_input.memory.layer_mem = 10691.0
        model_input.memory.static_mem0 = 3383.0
        model_input.memory.static_mem = 2382.0
        model_input.memory.lm_head_mem = 8675.0
        model_input.memory.mem_lim_stage0 = 51913.0
        model_input.memory.mem_lim_others = 52914.0
        model_input.memory.mem_lim_last = 46621.0
        model_input.memory.print_mem()
        cur_solution = pipeline_parallel.solve_problem(model_input)
        self.assertLessEqual(cur_solution.object_value, 1132.67)
        self.assertGreaterEqual(cur_solution.object_value, 1132.65)

    def test768_8k(self):
        yaml_file = self.get_yaml_path('layers62_micro160_dp24_tp8_pp4_vp1_ep32.yaml')
        mind_former_file = '~/mindformers/run_mindformer.py'
        expert_input = ExpertInput(yaml_file, mind_former_file)
        expert_input.solver_name = 'HIGHS'
        expert_input.is_dryrun = True
        expert_input.is_double_check = True
        expert_input.time_limit = 3 * 60
        expert_input.backward_ratio = 1.627
        expert_input.layer_ratio = 0.71
        expert_input.head_loss = 1.5
        expert_input.recompute_ratio = 0.71
        output_file = 'dryrun_output'
        pipeline_parallel.pp_calculator(expert_input)
        dryrun.all_rank_dryrun(yaml_file, mind_former_file, output_file)

    def test768_4k_gp(self):
        yaml_file = self.get_yaml_path('768die4k_gp.yaml')
        mind_former_file = '~/mindformers/run_mindformer.py'
        expert_input = ExpertInput(yaml_file, mind_former_file)
        expert_input.solver_name = 'HIGHS'
        expert_input.is_dryrun = False
        expert_input.is_double_check = False
        expert_input.time_limit = 1 * 60
        expert_input.layer_ratio = 0.71
        expert_input.backward_ratio = 1.627
        expert_input.recompute_ratio = 0.246
        expert_input.head_loss = 1.493
        output_file = 'dryrun_output_7684k_vp4'
        # pipeline_parallel.pp_calculator(expert_input)
        dryrun.all_rank_dryrun(yaml_file, mind_former_file, output_file)

    def test512_8k(self):
        yaml_file = self.get_yaml_path('layers62_micro960_dp4_tp8_pp16_vp1_ep32.yaml')
        mind_former_file = '~/mindformers/run_mindformer.py'
        expert_input = ExpertInput(yaml_file, mind_former_file)
        expert_input.solver_name = 'HIGHS'
        expert_input.is_dryrun = False
        expert_input.is_double_check = False
        expert_input.time_limit = 3 * 60
        expert_input.layer_ratio = 0.71
        expert_input.backward_ratio = 1.627
        expert_input.recompute_ratio = 0.246
        expert_input.head_loss = 1.493
        # model_input = InitConfig(yaml_file, mind_former_file, is_dryrun, layer_ratio, backward_ratio)
        output_file = 'dryrun_output_5128k_t1_vp2'
        # pipeline_parallel.pp_calculator(expert_input)
        dryrun.all_rank_dryrun(yaml_file, mind_former_file, output_file)

    def test512_8k_swap(self):
        yaml_file = self.get_yaml_path('512_8k_swap.yaml')
        mind_former_file = '~/mindformers/run_mindformer.py'
        expert_input = ExpertInput(yaml_file, mind_former_file)
        expert_input.solver_name = 'HIGHS'
        expert_input.is_dryrun = False
        expert_input.is_double_check = False
        expert_input.time_limit = 2 * 60
        expert_input.layer_ratio = 0.71
        expert_input.backward_ratio = 1.627
        expert_input.recompute_ratio = 0.246
        expert_input.head_loss = 1.493
        output_file = 'dryrun_output_5128k_swap_modify'
        # pipeline_parallel.pp_calculator(expert_input)
        dryrun.all_rank_dryrun(yaml_file, mind_former_file, output_file)

    def test512_8k_no_swap(self):
        yaml_file = self.get_yaml_path('512_8k_no_swap.yaml')
        mind_former_file = '~/mindformers/run_mindformer.py'
        expert_input = ExpertInput(yaml_file, mind_former_file)
        expert_input.solver_name = 'HIGHS'
        expert_input.is_dryrun = True
        expert_input.is_double_check = False
        expert_input.time_limit = 5 * 60
        expert_input.layer_ratio = 0.71
        expert_input.backward_ratio = 1.627
        expert_input.recompute_ratio = 0.246
        expert_input.head_loss = 1.493
        output_file = 'dryrun_output_5128k_no_swap'
        # pipeline_parallel.pp_calculator(expert_input)
        dryrun.all_rank_dryrun(yaml_file, mind_former_file, output_file)

    def test512_4k(self):
        yaml_file = self.get_yaml_path('layers62_micro60_dp32_tp4_pp4_vp1_ep128.yaml')
        mind_former_file = '~/mindformers/run_mindformer.py'
        expert_input = ExpertInput(yaml_file, mind_former_file)
        expert_input.solver_name = 'HIGHS'
        expert_input.is_dryrun = True
        expert_input.is_double_check = True
        expert_input.time_limit = 3 * 60
        expert_input.backward_ratio = 1.627
        expert_input.layer_ratio = 0.71
        expert_input.head_loss = 1.5
        expert_input.recompute_ratio = 0.71
        # model_input = InitConfig(yaml_file, mind_former_file, is_dryrun, layer_ratio, backward_ratio)
        output_file = 'dryrun_output'
        pipeline_parallel.pp_calculator(expert_input)
        dryrun.all_rank_dryrun(yaml_file, mind_former_file, output_file)


if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestSuite()
    suite.addTest(PipeTestCase('test_micro64_dp16_tp4_pp16_vp1_ep8'))
    runner = unittest.TextTestRunner()
    runner.run(suite)