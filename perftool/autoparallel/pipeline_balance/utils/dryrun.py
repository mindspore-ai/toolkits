#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
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
"""automatic dryrun"""

import os
import shutil
import subprocess
import re
import time

import toolkit.pipeline_balance.utils.recompute as Recompute
from toolkit.pipeline_balance.utils.logger import logger
from toolkit.pipeline_balance.utils.config import (
    convert_to_mf_format,
    parse_training_config,
    initialize_layer_json,
    ModelInfo,
)
from toolkit.pipeline_balance.utils.interactive import dryrun_guide
from toolkit.pipeline_balance.utils.stage import StagesBuilder
from toolkit.pipeline_balance.utils.compute_memory import ComputeMemory
from toolkit.pipeline_balance.utils.computation_analyzer import ComputationAnalyzer


DRYRUN_CONFIG_PATH = "./dryrun_configs"
RUN_MINDFORMER = "../../../run_mindformer.py"
DRYRUN_SCRIPT = "./dryrun.sh"
COMMAND_TEMPLATE = "python {run_mindformer} --config {config} --run_model train --use_parallel True"
TRAINCALLBACK = "  - type: TrainCallBack\n    stop_step: 2\n"
LLAMA_LAYER_LIST = {
    "pre_defined_layer": {
        "LlamaEmbedding": 0,
        "lm_head-Linear": -1,
        "LlamaRMSNorm": -1,
    },
    "auto_partition_layer": {"LLamaDecodeLayer": 1},
}


def dump_config(config_file: str, yaml_configs):
    """dump dryrun config"""
    dryrun_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), DRYRUN_CONFIG_PATH
    )
    os.makedirs(dryrun_config_path, exist_ok=True)

    dryrun_configs = []
    for i in range(len(yaml_configs)):
        dryrun_config = f"{dryrun_config_path}/round_{i}.yaml"
        dryrun_configs.append(dryrun_config)
        shutil.copy(config_file, dryrun_config)
        with open(dryrun_config, "r+") as f:
            lines = f.readlines()
            for line_idx, line in enumerate(lines):
                lines[line_idx] = re.sub(
                    r"offset:.*$",
                    f"{Recompute.OFFSET}: {yaml_configs[i][Recompute.OFFSET]}",
                    line,
                )
                for rec in [
                    Recompute.TYPE.FULL,
                    Recompute.TYPE.SLCT,
                    Recompute.TYPE.COMM,
                ]:
                    if re.match(rf"^(?:\s*){Recompute.YAML_NAME[rec]}:.*$", line):
                        lines[line_idx] = (
                            f"  {Recompute.YAML_NAME[rec]}: {yaml_configs[i][Recompute.YAML_NAME[rec]]}\n"
                        )
            f.seek(0)
            f.writelines(lines)
            f.truncate()
        with open(dryrun_config, "r") as f:
            lines = f.readlines()
            has_train_callback = False
            for i, line in enumerate(lines):
                if re.match("^(?:\s*)stop_step:.*$", line):
                    line = re.sub(r"^(?:\s*)stop_step:.*$", "stop_step: 2", line)
                    has_train_callback = True
                    break
            if not has_train_callback:
                for i, line in enumerate(lines):
                    if re.match(r"^(?:\s*)callbacks:.*$", line):
                        lines.insert(i + 1, TRAINCALLBACK)
        with open(dryrun_config, "w") as f:
            f.writelines(lines)

    return dryrun_configs


def gather_mem_results(dryrun_folders: str, pp: int, timeout=30):
    """gather memory results in given folder"""
    regex = re.compile("Used peak")
    t = 0
    while t < timeout:
        t += 1
        mem_results = []
        for dryrun_folder in dryrun_folders:
            mem_result = []
            files = [
                f
                for f in os.listdir(dryrun_folder)
                if os.path.isfile(os.path.join(dryrun_folder, f))
            ]
            sorted_files = sorted(
                files, key=lambda x: int(re.search(r"(\d+)", x).group(1))
            ) # sort files based on stage id
            for file in sorted_files:
                file_path = os.path.join(dryrun_folder, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, start=1):
                            if regex.search(line):
                                mem_result.append(
                                    int(re.search(r"(\d+)", line.strip()).group(1))
                                )
                except (UnicodeDecodeError, OSError):
                    # igonre unreadble files
                    pass
            mem_results.append(mem_result)
        if all(len(mem_result) == pp for mem_result in mem_results):
            logger.output(mem_results)
            return mem_results
        time.sleep(60)

    logger.error(
        "Failed to retrieve memory stats. Please check 1. your config file is correct and 2. your enviroment supports dryrun."
    )
    return None


def auto_dryrun(dryrun_configs, output_folder: str):
    """automatically dryrun"""
    if dryrun_configs is None:
        logger.error("dryrun configs cannot be NONE")
        return
    extracted_params = parse_training_config(dryrun_configs[0])
    dp = extracted_params["data_parallel"]
    mp = extracted_params["model_parallel"]
    pp = extracted_params["pipeline_stage"]
    rank_size = dp * mp * pp
    dryrun_sh = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), DRYRUN_SCRIPT
    )
    run_mindformer_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), RUN_MINDFORMER
    )
    dryrun_folders = []
    for i in range(len(dryrun_configs)):
        dryrun_folder = os.path.join(output_folder + f"round_{i}")
        if os.path.exists(dryrun_folder):
            shutil.rmtree(dryrun_folder)
        os.makedirs(dryrun_folder, exist_ok=True)
        subprocess.run(
            [
                "bash",
                dryrun_sh,
                COMMAND_TEMPLATE.format(run_mindformer=run_mindformer_path, config=dryrun_configs[i]),
                f"{rank_size}",
                f"{pp}",
                dryrun_folder,
            ]
        )
        dryrun_folders.append(dryrun_folder)

    return dryrun_folders


def convert_auto_to_sapp(yaml_configs, mem_results, num_layers, timeline_folder, model_name):
    """convert auto ppb results to sapp interface"""
    num_stages = len(mem_results[0])
    rounds = len(mem_results)
    head_mem = mem_results[0][0]
    tail_mem = mem_results[0][-1]
    body_mems = []
    for mem_result in mem_results:
        body_mem = mem_result[1:-1]
        body_mems.append(body_mem)

    offset_config = [config[Recompute.OFFSET] for config in yaml_configs]
    recompute_config = {}
    for rec in [
        Recompute.YAML_NAME[Recompute.TYPE.FULL],
        Recompute.YAML_NAME[Recompute.TYPE.SLCT],
        Recompute.YAML_NAME[Recompute.TYPE.COMM],
    ]:
        rec_list = []
        for config in yaml_configs:
            rec_list.append(config[rec])
        recompute_config.update({rec:rec_list})
    stage_ids = [[i for i in range(1, num_stages-1)] for _ in range(rounds)]

    if rounds == 1: # squeeze 2d list to 1d
        offset_config = offset_config[0]
        recompute_config = {key: value[0] for key, value in recompute_config.items()}
        stage_ids = stage_ids[0]
        body_mems = body_mems[0]
    stage_builder = StagesBuilder(
        num_stages, num_layers, recompute_config, offset_config, 
        stage_ids, head_mem, tail_mem, body_mems
    )
    stages = stage_builder.build_stages()
    comp_mem = ComputeMemory(num_stages, stages)

    head_time, body_time, tail_time = 0, 0, 0
    LLAMA_LAYER_LIST["auto_partition_layer"].update({"LLamaDecodeLayer": num_layers})
    analyzer = ComputationAnalyzer(timeline_folder, model_name, 8,LLAMA_LAYER_LIST)
    cost_list = analyzer.layer_with_cost_list
    for layer, time in cost_list.items():
        if layer in LLAMA_LAYER_LIST["pre_defined_layer"] and LLAMA_LAYER_LIST["pre_defined_layer"][layer] == 0:
            head_time += time
        elif layer in LLAMA_LAYER_LIST["pre_defined_layer"] and LLAMA_LAYER_LIST["pre_defined_layer"][layer] == -1:
            tail_time += time
        else:
            body_time += time

    mi = ModelInfo(model_name, head_time, body_time, tail_time, num_layers)
    layer_json = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../layers"
    )
    initialize_layer_json(mi, comp_mem, layer_json)


def auto_ppb(config_path: str, timeline_folder: str, model_name: str):
    extracted_params = parse_training_config(config_path)
    offset_config_list, rec_config_list = dryrun_guide(extracted_params)
    num_layers = extracted_params['num_layers']
    pipeline_stage = extracted_params['pipeline_stage']
    yaml_configs = convert_to_mf_format(offset_config_list, rec_config_list)
    configs = dump_config(config_path, yaml_configs)
    dryrun_folders = auto_dryrun(configs, "./output/")
    mem_results = gather_mem_results(dryrun_folders, pipeline_stage)
    convert_auto_to_sapp(yaml_configs, mem_results, num_layers, timeline_folder, model_name)