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
"""run pipeline balance"""
import os
import sys
import argparse

from toolkit.pipeline_balance.utils.logger import logger
from toolkit.pipeline_balance.utils.config import (
    convert_init_to_sapp, get_stage_const_mem, parse_training_config,
    time_parser_auto, auto_ppb_config_parser
)
from toolkit.pipeline_balance.utils.layer import generate_layers_list
from toolkit.pipeline_balance.sapp.sapp_pipeline import SappPipeline
from toolkit.pipeline_balance.utils.dryrun import auto_ppb, auto_validate
import toolkit.pipeline_balance.utils.interactive as Interactive

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='SAPP AutoBalancing', description=(
                'Balance layers onto pipeline stages, '
                + 'considering recomputation and interleaving'),
                                     epilog='')

    # Pipeline info
    parser.add_argument('-s', '--stage', type=int, default=4, help="Number of stages")
    parser.add_argument('-mb', '--micro_batch', type=int, default=4, help="Number of micro batch")
    parser.add_argument('-i', '--interleave_degree', type=int, default=1, help="Interleave level")

    # Memory size
    parser.add_argument('-mem', '--max_memory', type=int, default=56000,
                        help="Maximum memory available (MB)")
    # vpp schedule
    parser.add_argument('-lm', '--less_memory',
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False,
                        help="Compute Memory with 'Less Memory interleave' option")

    parser.add_argument('-o', '--output_folder', type=str, default="./output",
                        help="output files location")

    # Model info
    parser.add_argument('-m', '--model_name', type=str, default="model_name", help="")

    # Search time
    parser.add_argument('-t', '--time_limit', type=int, default=90,
                        help="Limitation on searching time")

    # Optimization level
    parser.add_argument('-O', '--optimization_level', type=int, default=1,
                        help="Defines optimization level when Stage (S) = Micro Batch number (M)."
                        + " 0 for same approach as M > S. "
                        + " 1 (default) generally better. "
                        + " 2 better for memory constrained cases. ")

    # Simulate naive or manual config
    parser.add_argument('-naive', '--simulate_naive', type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        default=False,
                        help="Simualte naive configs")
    parser.add_argument('-manual', '--manual_config', type=str, default=None,
                        help="Path of manual config")

    # Layer info
    parser.add_argument('-lf', '--layer_json', type=str, default="./layers/llama.json",
                        help="Path to the layer json file")
    parser.add_argument('-dump', '--dump_layer',  # type=bool,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False,
                        help="Dump the layers")

    # For Initialization
    parser.add_argument('-init', '--init', type=str, default=None,
                        help="Path to the init file")

    # Computation argument
    parser.add_argument('-exec', '--exec',  # type=bool,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True,
                        help="Compute solver")

    # Auto dryrun
    parser.add_argument('-auto', '--auto_ppb', action='store_true', help="Fully automate pipeline balance")
    # Training Yaml configuration
    parser.add_argument('-train', '--training_config', type=str, default=None,
                        help="Path of training config (.ymal)")
    # Seq pipe config
    parser.add_argument('-seq', '--seq', type=int, default=1,
                        help="Sequence chunk split number")


    args = parser.parse_args()

    if len(sys.argv) == 1:
        Interactive.main()
        sys.exit(0)

    layer_json = args.layer_json
    model_name = args.model_name
    time_limit = args.time_limit
    less_memory = args.less_memory

    output_folder = os.path.join(os.path.dirname(__file__), args.output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)



    extracted_training_params = None
    if args.training_config:
        if os.path.isabs(args.training_config):
            training_config_path = args.training_config
        else:
            training_config_path = os.path.join(os.path.dirname(__file__), args.training_config)

        if training_config_path.endswith('yaml') or training_config_path.endswith('yml'):
            extracted_training_params = parse_training_config(training_config_path)
            logger.output("Extracted training parameters:")
            for key, value in extracted_training_params.items():
                logger.output("%s: %s", key, value)
            number_of_stage = extracted_training_params["pipeline_stage"]
            number_of_micro_batch = extracted_training_params["micro_batch_num"]
        else:
            logger.error("Training config file must be a YAML file")

    if args.micro_batch:
        number_of_micro_batch = args.micro_batch

    if args.stage:
        number_of_stage = args.stage


    config_file = os.path.join(os.path.dirname(__file__), "cfgs/auto_ppb_config.yaml") 
    auto_ppb_config = auto_ppb_config_parser(config_file)
    auto_ppb_config["model_name"] = model_name
    training_config_path = auto_ppb_config["training_config"]["training_config_path"]
    if args.auto_ppb:
        layer_times = time_parser_auto(config_file) 
        auto_ppb_config["layer_times"] = layer_times
        auto_ppb(training_config_path, auto_ppb_config, output_folder)
        # auto_ppb(training_config_path, model_name)
        layer_json = os.path.join(output_folder, "../layers", model_name + ".json")

    seq_split_num = args.seq

    #seq_chunk < 4k
    if seq_split_num > 1:
        if extracted_training_params['seq_length'] is not None:
            if extracted_training_params['seq_length'] / seq_split_num < 4096:
                logger.warning(f"Seq_chunk_length will < 4k after splitting with seq_split_num {seq_split_num}, \
                               which will affect performance")
                if seq_split_num == 2:
                    seq_split_num = 1
                    logger.warning("seqpipe has deactivated (seq_split_num = 1)")
                else:
                    seq_split_num = extracted_training_params['seq_length'] // 4096
                    logger.warning(f"To avoid affecting performance, seq_split_num has updated as {seq_split_num}")

    if not args.auto_ppb and args.init:
        init_file = os.path.join(os.path.dirname(__file__), args.init)  
        convert_init_to_sapp(model_name, init_file, training_config_path)


    manual_config = None
    if args.manual_config:
        manual_config = os.path.join(os.path.dirname(__file__), args.manual_config)
        manual_config = None if (not manual_config.endswith('yaml') and
                                 not manual_config.endswith('yml')) else manual_config

    max_memory = args.max_memory
    interleave_degree = args.interleave_degree
    optimization_level = args.optimization_level

    layers = generate_layers_list(layer_json)
    constant_memory = get_stage_const_mem(layer_json)

    for layer in layers:
        logger.output("%s", layer)

    if args.dump_layer:
        for layer in layers:
            layer.dump()

    if args.exec:
        pipe = SappPipeline(model_name=model_name, num_of_stage=number_of_stage,
                            num_of_micro_batch=number_of_micro_batch, max_memory=max_memory,
                            layers=layers, num_of_interleave=interleave_degree,
                            vpp_less_memory=less_memory, constant_memory=constant_memory,
                            optimization_level=optimization_level, extracted_training_params=extracted_training_params,
                            seq_split_num=seq_split_num)

        pipe.construct_problem(solver="pulp")
        pipe.solve_problem(time_limit=time_limit, dump_folder=output_folder)
        pipe.print_yaml_results()

        result = str(model_name)
        result += "_" + str(max_memory)
        result += "_" + str(interleave_degree)
        result += "_" + str(number_of_stage)
        result += "_" + str(number_of_micro_batch)
        result += "_" + str(optimization_level)
        result += "_" + str(time_limit)
        result += "_" + str(less_memory)
        result += ".svg"


        if args.simulate_naive:
            logger.output("Simulating naive configs")
            pipe.simulate_naive(layers, output_folder)
        if manual_config:
            logger.output("Simulating manual configs")
            pipe.simulate_file(manual_config, output_folder)
        else:
            total_time = pipe.simulate(show=True, file_name=os.path.join(output_folder, result))
            logger.output("total_time: %d", total_time)
            logger.output("time: %s", pipe.get_time())
            logger.output("mem_par: %s", pipe.get_memory_parameter())
            logger.output("mem_act: %s", pipe.get_memory_activation())

        if auto_ppb_config["validate"]:
            validate_settings = {"pp_interleave_num": interleave_degree, "lm": less_memory, "micro_batch_num": number_of_micro_batch,
                                 "pipeline_stage": number_of_stage}
            auto_validate(training_config_path, auto_ppb_config, validate_settings,
                          pipe.get_yaml_results(), max_memory,
                          pipe.problem_.mem_estimate_, pipe.mem_simulate, output_folder)
