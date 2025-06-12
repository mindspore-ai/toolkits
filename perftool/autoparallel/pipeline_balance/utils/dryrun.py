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
import copy
import yaml


import toolkit.pipeline_balance.utils.recompute as Recompute
from toolkit.pipeline_balance.utils.logger import logger
from toolkit.pipeline_balance.utils.config import (
    convert_to_mf_format,
    parse_training_config,
    initialize_layer_json,
    check_mem_results,
    compare_mem_results,
    ModelInfo,
)
from toolkit.pipeline_balance.utils.interactive import dryrun_guide
from toolkit.pipeline_balance.utils.stage import StagesBuilder
from toolkit.pipeline_balance.utils.compute_memory import ComputeMemory
from toolkit.pipeline_balance.utils.computation_analyzer import ComputationAnalyzer


DRYRUN_CONFIG_PATH = "./dryrun_configs"
RUN_MINDFORMER = "../../../run_mindformer.py"
DRYRUN_SCRIPT = "./dryrun.sh"
COMMAND_TEMPLATE = "python {run_mindformer} --config {config} --register_path research/deepseek3 --run_model train --use_parallel True"
TRAINCALLBACK = "  - type: TrainCallBack\n    stop_step: 2\n"
LLAMA_LAYER_LIST = {
    "pre_defined_layer": {
        "LlamaEmbedding": 0,
        "lm_head-Linear": -1,
        "LlamaRMSNorm": -1,
    },
    "auto_partition_layer": {"LLamaDecodeLayer": 1},
}


def dump_config(config_file: str, yaml_configs, output_path: str, validate_config=None):
    """dump dryrun config"""
    os.makedirs(output_path, exist_ok=True)
    dryrun_configs = []
    for i in range(len(yaml_configs)):
        dryrun_config = f"{output_path}/round_{i}.yaml"
        dryrun_configs.append(dryrun_config)
        shutil.copy(config_file, dryrun_config)
        # 
        if(validate_config):
            update_validate_settings(dryrun_config, validate_config)
        else:
            delete_interleave_settings(dryrun_config)
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




def delete_interleave_settings(config_path):
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    in_pipeline_config = False
    parent_indent = 0
    
    for line in lines:
        current_indent = len(line) - len(line.lstrip())
        
        # Detect pipeline_config start
        if line.strip() == 'pipeline_config:':
            in_pipeline_config = True
            parent_indent = current_indent
            continue
        
        # Skip indented content under pipeline_config
        if in_pipeline_config:
            if current_indent > parent_indent:
                continue
            in_pipeline_config = False  # Exit pipeline_config block
        
        # Skip pp_interleave_num leaf
        if line.strip().startswith('pp_interleave_num:'):
            continue
        
        new_lines.append(line)
    
    with open(config_path, 'w') as f:
        f.writelines(new_lines)


def update_validate_settings(config_path, validate_settings):
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    needs_pipeline_config = True
    settings_require = {"pipeline_interleave": True, 'pp_interleave_num': True,
                        'micro_batch_num': True, 'pipeline_scheduler': True,
                        'pipeline_stage': True}

    validate_settings["pipeline_interleave"] = True
    validate_settings["pipeline_scheduler"] = "1f1b"

    # First pass: ensure pipeline_config exists and collect other info
    for idx, line in enumerate(lines):
        if line.strip() == 'pipeline_config:':
            needs_pipeline_config = False
        else:
            key = line.split(':')[0].strip()
            if key in settings_require:
                lines[idx] = re.sub(
                    rf'({key}:\s*)(&\w+\s+)?\d+',
                    lambda m: f"{m.group(1)}{m.group(2) or ''}{validate_settings[key]}",
                    line
                )
                settings_require[key] = False

    
    # Add pipeline_config if missing
    if needs_pipeline_config:
        for i, line in enumerate(lines):
            if line.strip() == 'parallel:':
                indent = len(line) - len(line.lstrip())
                lines.insert(i+1, 
                    f"{' '*(indent+2)}pipeline_config:\n"
                    f"{' '*(indent+4)}pipeline_interleave: True\n"
                    f"{' '*(indent+4)}pipeline_scheduler: '1f1b'\n")
                
                settings_require["pipeline_interleave"] = False
                settings_require["pipeline_scheduler"] = False
                break


    # Add all required leaf nodes
    for i, line in enumerate(lines):
        if line.strip() == 'pipeline_config:':
            idx = i
            indent = len(line) - len(line.lstrip())
            if settings_require["pipeline_interleave"]:
                idx += 1
                lines.insert(idx, 
                    f"{' '*(indent+2)}pipeline_interleave: True\n")
            if settings_require["pipeline_scheduler"]:
                idx += 1
                lines.insert(idx, 
                    f"{' '*(indent+2)}pipeline_scheduler: '1f1b'\n")
        elif line.strip() == 'parallel_config:':
            indent = len(line) - len(line.lstrip())
            idx = i
            if(settings_require["micro_batch_num"]):
                idx += 1
                lines.insert(idx, f"{' '*(indent+2)}micro_batch_num: {validate_settings['micro_batch_num']}\n")
            if(settings_require["pipeline_stage"]):
                idx += 1
                lines.insert(idx, f"{' '*(indent+2)}pipeline_stage: {validate_settings['pipeline_stage']}\n")
        elif line.strip() == 'model_config:' and settings_require["pp_interleave_num"]:
            indent = len(line) - len(line.lstrip())
            lines.insert(i+1, f"{' '*(indent+2)}pp_interleave_num: {validate_settings['pp_interleave_num']}\n")
    
    with open(config_path, 'w') as f:
        f.writelines(lines)


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
                mem = -1
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, start=1):
                            if regex.search(line):
                                mem = int(re.search(r"(\d+)", line.strip()).group(1))
                except (UnicodeDecodeError, OSError):
                    # igonre unreadble files
                    pass
                if mem != -1:
                    mem_result.append(mem)
            mem_results.append(mem_result)
        if all(len(mem_result) == pp for mem_result in mem_results):
            logger.output(f"Gathered memories: {mem_results}")
            return mem_results
        time.sleep(60)

    logger.error(
        "Failed to retrieve memory stats. Please check 1. your config file is correct and 2. your enviroment supports dryrun."
    )
    return None


def auto_dryrun(dryrun_configs, output_folder: str, validate_settings=None):
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
        dryrun_folder = os.path.join(output_folder, f"round_{i}")
        if os.path.exists(dryrun_folder):
            shutil.rmtree(dryrun_folder)
        os.makedirs(dryrun_folder, exist_ok=True)
        if validate_settings:
            vpp_str = '1' if validate_settings["lm"] else '0'
            os.environ["ENABLE_LESS_MEM_VPP"] = vpp_str
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




def get_comp_mem(mem_results, yaml_configs, num_layers):
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

    return comp_mem






def convert_auto_to_sapp(yaml_configs, mem_results_list, layer_times, num_layers, model_name, output_folder):
    """convert auto ppb results to sapp interface"""

    comp_mem_list = []
    for idx, mem_results in enumerate(mem_results_list):
        comp_mem_list.append(get_comp_mem(mem_results, yaml_configs, num_layers[idx]))

    # head_time, body_time, tail_time = 18, 10, 25
    head_time, body_time, tail_time = layer_times
    # DeekSeekV2DecodeLayer
    body_layers_names = [e[0] for e in next(iter(body_time.values()))]
    mi = ModelInfo(model_name, head_time, body_time, tail_time, num_layers)
    layer_json = os.path.join(
        output_folder, "../layers"
    )

    initialize_layer_json(mi, comp_mem_list, body_layers_names, layer_json)


def auto_ppb(config_path: str, auto_ppb_config, output_folder: str):
    model_name = auto_ppb_config["model_name"]
    layer_times = auto_ppb_config["layer_times"]
    extracted_params = parse_training_config(config_path)
    offset_config_list, rec_config_list = dryrun_guide(extracted_params, auto_ppb_config["model_type"])

    if auto_ppb_config["model_type"] == "deepseek":
        # TODO: num_layers = auto_ppb_config["training_config"]["body_layer_nums"]
        num_layers = [extracted_params['num_layers'] - 3] # for deepseek

    else:
        num_layers = [extracted_params['num_layers']]
    pipeline_stage = extracted_params['pipeline_stage']
    yaml_configs_dryrun = convert_to_mf_format(offset_config_list, rec_config_list)
    yaml_configs_compute = copy.deepcopy(yaml_configs_dryrun)
    if auto_ppb_config["model_type"] == "deepseek":
        for i in range(len(yaml_configs_dryrun)): # for deepseek
            for _, value in yaml_configs_dryrun[i].items():
                value[0] += 3
    dryrun_config_path = os.path.join(
        output_folder, DRYRUN_CONFIG_PATH
    )
    configs = dump_config(config_path, yaml_configs_dryrun, dryrun_config_path)
    log_path = os.path.join(output_folder, "dryrun_log")
    dryrun_folders = auto_dryrun(configs, log_path)
    mem_results = gather_mem_results(dryrun_folders, pipeline_stage)
    mem_results_list = [mem_results]
    convert_auto_to_sapp(yaml_configs_compute, mem_results_list, layer_times, num_layers, model_name, output_folder)


def auto_validate(config_path: str, auto_ppb_config, validate_settings, sapp_result, max_memory,
                  mem_estimate, mem_simulate, output_folder: str):
    extracted_params = parse_training_config(config_path)
    sapp_configs_validate = [next(iter(sapp_result.values()))]
    if auto_ppb_config["model_type"] == "deepseek":
        for _, value in sapp_configs_validate[0].items(): # for deepseek
            if isinstance(value[0], list): #interleave
                value[0][0] += 3
                value[-1][-1] += 1
            else:
                value[0] += 3
                value[-1] += 1
    logger.info(f"sapp_configs_validate: {sapp_configs_validate}")

    validate_config_path = os.path.join(
        output_folder, "validate_config"
    )
    configs = dump_config(config_path, sapp_configs_validate, validate_config_path, validate_settings)
    log_path = os.path.join(output_folder, "validate_log")
    dryrun_folders = auto_dryrun(configs, log_path, validate_settings)
    mem_results = gather_mem_results(dryrun_folders, extracted_params['pipeline_stage'])[0]
    if not check_mem_results(mem_results, max_memory):
        logger.warning(f"Actual peak memory exceeds the max memory: {max_memory} in validation")
    compare_mem_results(mem_estimate, mem_results, "Memory Estimated", "Actual Costs")
    compare_mem_results(mem_simulate, mem_results, "Memory Simulated", "Actual Costs")
