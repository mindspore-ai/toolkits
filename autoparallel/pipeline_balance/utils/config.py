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
"""config json generator"""
import json
import os
from dataclasses import dataclass, asdict
from math import ceil
import random

import yaml
import numpy as np

from toolkit.pipeline_balance.utils.check_rules import check_yaml_depth_before_loading
from toolkit.pipeline_balance.utils.logger import logger
from toolkit.pipeline_balance.sapp.sapp_solver import SappSolver
from toolkit.pipeline_balance.utils.layer import Layer
from toolkit.pipeline_balance.utils.compute_memory import ComputeMemory
from toolkit.pipeline_balance.utils.stage import StagesBuilder
import toolkit.pipeline_balance.utils.recompute as Recompute
from toolkit.pipeline_balance.utils.computation_analyzer import ComputationAnalyzer

random.seed()


@dataclass
class LayersDescription:
    """layers description"""

    name: str
    type: Layer.type_enum
    model_name: str
    time: int
    nb_layer: int
    memory_parameter: int

    def __init__(
            self, layer_type: Layer.type_enum, time: int, nb_layer: int, model_name: str
    ):
        self.type = layer_type.name
        self.time = time
        self.name = layer_type.name
        self.nb_layer = nb_layer
        self.model_name = model_name


@dataclass
class ModelInfo:
    """basic info of a model"""

    name: str
    stage_const_mem: int
    layers_description: list[LayersDescription]

    def __init__(self, model_name, head_time, body_time, tail_time, nb_layer):
        self.name = model_name
        self.stage_const_mem = 0
        self.layers_description = []
        self.layers_description.append(
            LayersDescription(Layer.type_enum.HEAD, head_time, 1, model_name)
        )
        self.layers_description.append(
            LayersDescription(Layer.type_enum.BODY, body_time, nb_layer, model_name)
        )
        self.layers_description.append(
            LayersDescription(Layer.type_enum.TAIL, tail_time, 1, model_name)
        )

    def get_layer_by_type(self, layer_type: Layer.type_enum):
        for layer in self.layers_description:
            if layer.type == layer_type.name:
                return layer
        return None

    def set_stage_const_mem(self, mem_const):
        self.stage_const_mem = mem_const

    def layer_memory_update(self, mem_act, mem_par, mem_head, mem_tail):
        """update input memories to layer description"""
        self.get_layer_by_type(Layer.type_enum.HEAD).memory_parameter = mem_head
        self.get_layer_by_type(Layer.type_enum.TAIL).memory_parameter = mem_tail
        self.get_layer_by_type(Layer.type_enum.BODY).memory_parameter = mem_par

        json_data = asdict(self)

        for rec in Recompute.TYPE:
            rec_mem = mem_act.get(rec)
            if rec_mem is None:
                continue
            for layer in json_data["layers_description"]:
                if layer["type"] == Layer.type_enum.BODY.name:
                    layer[Recompute.JSON_MEMORY_NAME[rec]] = rec_mem
        self.to_json_ = json_data

    def dump_json(self, file_name):
        with open(file_name, "w+") as json_file:
            json.dump(self.to_json_, json_file, indent=4)


def time_parser(file_name: str, model_name: str):
    """parse time given by yaml"""
    if file_name is None:
        logger.error("input file cannot be none")
        raise ValueError("input file cannot be none")

    if not file_name.endswith("yaml") and not file_name.endswith("yml"):
        logger.error("Only accept yaml as input format")
        raise ValueError(f"Only accept yaml as input format. not {file_name}")

    filepath = os.path.realpath(file_name)
    with open(filepath, encoding="utf-8") as fp:
        check_yaml_depth_before_loading(fp)
        fp.seek(0)
        cfg_dict = yaml.safe_load(fp)

    head_time = 0
    body_time = 0
    tail_time = 0

    if "time_config" in cfg_dict:
        head_time = cfg_dict["time_config"].get("head")
        body_time = cfg_dict["time_config"].get("body")
        tail_time = cfg_dict["time_config"].get("tail")
        if all(key in cfg_dict["time_config"] for key in ["head", "body", "tail"]):
            return head_time, body_time, tail_time

    if cfg_dict.get("profiling_config"):
        head_layers = cfg_dict["profiling_config"].get("head_layers", ["LlamaEmbedding"])
        body_layers = cfg_dict["profiling_config"].get("body_layers", ["LLamaDecodeLayer"])
        tail_layers = cfg_dict["profiling_config"].get("tail_layers", ["lm_head-Linear", "LlamaRMSNorm"])
        if isinstance(head_layers, str):
            head_layers = [head_layers]
        if isinstance(tail_layers, str):
            tail_layers = [tail_layers]
        if isinstance(body_layers, str):
            body_layers = [body_layers]

        num_layer = cfg_dict["pipeline_config"]["num_layer"]
        micro_batch_num = cfg_dict["profiling_config"]["micro_batch_num"]
        timeline_folder_path = cfg_dict["profiling_config"]["folder_path"]
        layer_list = {"pre_defined_layer": {}, "auto_partition_layer": {}}
        for layer in head_layers:
            layer_list["pre_defined_layer"].update({layer: 0})
        for layer in tail_layers:
            layer_list["pre_defined_layer"].update({layer: -1})
        for layer in body_layers:
            layer_list["auto_partition_layer"].update({layer: num_layer})
        analyzer = ComputationAnalyzer(timeline_folder_path, model_name, micro_batch_num, layer_list)
        cost_list = analyzer.layer_with_cost_list
        logger.info(cost_list)
        for layer, time in cost_list.items():
            if layer in layer_list["pre_defined_layer"] and layer_list["pre_defined_layer"][layer] == 0:
                head_time += time
            elif layer in layer_list["pre_defined_layer"] and layer_list["pre_defined_layer"][layer] == -1:
                tail_time += time
            else:
                body_time += time

    logger.info(f"head_time: {head_time}, body_time: {body_time}, tail_time: {tail_time}")

    return head_time, body_time, tail_time


def memory_parser(file_name: str):
    """parse input given by yaml"""
    if file_name is None:
        logger.error("input file cannot be none")
        raise ValueError("input file cannot be none")
    if not file_name.endswith("yaml") and not file_name.endswith("yml"):
        logger.error("Only accept yaml as input format")
        raise ValueError(f"Only accept yaml as input format. not {file_name}")

    filepath = os.path.realpath(file_name)
    with open(filepath, encoding="utf-8") as fp:
        check_yaml_depth_before_loading(fp)
        fp.seek(0)
        cfg_dict = yaml.safe_load(fp)

    # get pipeline config
    pipeline_num = cfg_dict["pipeline_config"]["pipeline_num"]
    num_layer = cfg_dict["pipeline_config"]["num_layer"]
    offset = cfg_dict["pipeline_config"]["offset"]
    recompute_config = cfg_dict["recompute_config"]
    # get memory usage
    stage_ids = cfg_dict["memory_usage"]["body_memories"]["stage_id"]
    mem_head_stage = cfg_dict["memory_usage"]["head_memory"]
    mem_tail_stage = cfg_dict["memory_usage"]["tail_memory"]
    body_memories = cfg_dict["memory_usage"]["body_memories"]["memories"]

    stage_builder = StagesBuilder(
        pipeline_num,
        num_layer,
        recompute_config,
        offset,
        stage_ids,
        mem_head_stage,
        mem_tail_stage,
        body_memories,
    )
    stages_a = stage_builder.build_stages()

    return pipeline_num, stages_a, num_layer


def initialize_layer_json(model_info: ModelInfo, comp_mem: ComputeMemory, output_folder: str = "./layers" ):
    """initialize layer description json file"""
    mem_act = {}
    for r in Recompute.TYPE:
        if comp_mem.recompute_considered_[r]:
            mem_act[r] = int(comp_mem.get_memory_activation(r))
            logger.info(
                "[INFO] %s = %f", Recompute.JSON_MEMORY_NAME[r],
                int(comp_mem.get_memory_activation(r))
            )
    if comp_mem.get_memory_const() is not None:
        mem_const = int(comp_mem.get_memory_const())
        logger.info(f"[INFO] memory_const       = {mem_const}")
        model_info.set_stage_const_mem(mem_const)
    mem_par = int(comp_mem.get_memory_parameter())
    mem_tail = int(comp_mem.get_memory_tail())
    mem_head = int(comp_mem.get_memory_head())
    logger.info(f"[INFO] memory_parameter  = {mem_par}")
    logger.info(f"[INFO] memory_tail       = {mem_tail}")
    logger.info(f"[INFO] memory_head       = {mem_head}")

    model_info.layer_memory_update(mem_act, mem_par, mem_head, mem_tail)
    model_info.dump_json(os.path.join(output_folder, model_info.name + ".json"))


def convert_init_to_sapp(model_name: str, file_name: str, output_folder: str = "./layer"):
    """convert init file to sapp interface json file"""
    num_stage, stages_a, num_layer = memory_parser(file_name)
    head_time, body_time, tail_time = time_parser(file_name, model_name)
    comp_mem = ComputeMemory(number_of_stage=num_stage, stages_A=stages_a)
    mi = ModelInfo(model_name, head_time, body_time, tail_time, num_layer)
    initialize_layer_json(mi, comp_mem, output_folder)


def get_stage_const_mem(layer_folder: str, model_name: str):
    """ Get stage constant memory from layer_folder/model_name.json"""
    json_layer = os.path.join(layer_folder, model_name + '.json')
    with open(json_layer, encoding="utf-8") as json_file:
        json_data = json.load(json_file)
        if "stage_const_mem" in json_data:
            return json_data["stage_const_mem"]
    return 0


def _generate_offset_config(rounds, unknowns, target_sum, array_length):
    """Generate legal random offset arrays"""
    while True:
        offset_config_list = []
        flat = []
        for _ in range(rounds):
            offset_config = _generate_offset_array(target_sum, array_length)
            offset_config_slice = offset_config[1:]
            offset_config_slice = offset_config_slice[:-1]
            flat.append(offset_config_slice)
            offset_config_list.append(offset_config)
        flat = np.array(flat)
        flat = flat.flatten()[0 : unknowns + 1]
        if not np.all(flat == flat[0]):
            return offset_config_list


def _generate_offset_array(target_sum, array_length):
    """Generate a random offset array"""
    if target_sum == array_length:
        return [0] * array_length

    if target_sum < array_length:
        logger.error("number of layers must be larger than stage number")
        return None

    random_array = np.random.randint(1, 10, size=array_length)
    total_sum = random_array.sum()
    scaled_array = (random_array / total_sum) * target_sum
    scaled_array = np.round(scaled_array).astype(int)
    scaled_array = np.maximum(scaled_array, 1)
    current_sum = scaled_array.sum()
    diff = target_sum - current_sum

    if diff >= 0:
        for i in range(abs(diff)):
            scaled_array[i] += 1
    else:
        count = 0
        for i in range(len(scaled_array)):
            # backwards iteration to avoid infinite loop
            if scaled_array[-1 - i] > 1:
                scaled_array[-1 - i] -= 1
                count += 1
                if count == abs(diff):
                    break

    baseline = target_sum // array_length
    offset = scaled_array - baseline
    return offset.tolist()


def _get_coef_matrix(pp, layer_per_stage, offset_config_list, rec_config_list, considered_rec):
    """get coef matrix of equations"""
    activation_nums = SappSolver.compute_activation_nums(pp, 1, 0)[0]
    coef_matrix = []
    rounds = len(offset_config_list)
    for round_ in range(rounds):
        for stage in range(pp):
            if stage not in [0, pp - 1]:
                coef_matrix.append(
                    [1, layer_per_stage + offset_config_list[round_][stage]]
                    + Recompute.to_list(
                        {
                            rec: rec_config_list[round_][rec][stage]
                                 * activation_nums[stage]
                            for rec in considered_rec
                        }
                    )
                )
            if len(coef_matrix) == 2 + len(considered_rec):
                return coef_matrix
    return None


def convert_to_mf_format(offset_config_list, rec_config_list):
    """convert result to mf format"""
    yaml_configs = []
    for round_, offset_config in enumerate(offset_config_list):
        yaml_config = {
            Recompute.OFFSET: [],
            Recompute.YAML_NAME[Recompute.TYPE.FULL]: [],
            Recompute.YAML_NAME[Recompute.TYPE.SLCT]: [],
            Recompute.YAML_NAME[Recompute.TYPE.COMM]: [],
        }
        yaml_config[Recompute.OFFSET] = offset_config
        pp = len(offset_config)
        slct = rec_config_list[round_].get(Recompute.TYPE.SLCT, [0] * pp)
        comm = rec_config_list[round_].get(Recompute.TYPE.COMM, [0] * pp)
        full = rec_config_list[round_].get(Recompute.TYPE.FULL, [0] * pp)
        both = rec_config_list[round_].get(Recompute.TYPE.BOTH, [0] * pp)

        yaml_config[Recompute.YAML_NAME[Recompute.TYPE.FULL]] = full
        yaml_config[Recompute.YAML_NAME[Recompute.TYPE.SLCT]] = [
            x + y + z for x, y, z in zip(slct, both, full)
        ]
        yaml_config[Recompute.YAML_NAME[Recompute.TYPE.COMM]] = [
            x + y + z for x, y, z in zip(comm, both, full)
        ]
        yaml_configs.append(yaml_config)
    return yaml_configs


def print_dryrun_config(yaml_configs):
    """print generated config"""
    logger.output(
        f"Please dryrun following config, {len(yaml_configs)} round(s) is needed"
    )
    for round_ in range(len(yaml_configs)):
        yaml_config = yaml_configs[round_]
        yaml_results = f"for round {round_ + 1}, please dryrun config:"
        for y, v in yaml_config.items():
            yaml_results += f"\n\t{y}: {v}"
        logger.output(yaml_results)


def generate_solvable_config(pp: int, num_layers: int, considered_rec: list):
    """generate a config that meets solvable criteria"""
    if pp == 2:
        logger.error("pp = 2 is not supported yet")
        return None

    considered_rec.append(Recompute.TYPE.NONE)
    rounds = ceil((2 + len(considered_rec)) / (pp - 2))
    layer_per_stage = num_layers // pp
    is_solvable = False
    offset_config_list = _generate_offset_config(
        rounds, 2 + len(considered_rec), num_layers, pp
    )
    while not is_solvable:
        rec_config_list = []
        for round_ in range(rounds):
            offset_config = offset_config_list[round_]
            layer_per_recompute = {r: [0] * pp for r in considered_rec}
            for rec in considered_rec:
                stage_sum = [
                    sum(col) for col in zip(*layer_per_recompute.values())
                ]  # summation of each rec in each stage
                layers_left = [
                    offset_config[i] + layer_per_stage - stage_sum[i] for i in range(pp)
                ]
                layer_per_recompute[rec] = [
                    random.randint(0, layers_left[i]) for i in range(pp)
                ]
            rec_config_list.append(layer_per_recompute)

        coef_matrix = _get_coef_matrix(
            pp, layer_per_stage, offset_config_list, rec_config_list, considered_rec
        )
        coef_rank = np.linalg.matrix_rank(coef_matrix)
        if coef_rank == len(considered_rec) + 2:
            is_solvable = True

    return offset_config_list, rec_config_list


def parse_training_config(yaml_path):
    """extract useful configuration from training yaml to calculate operator shape"""
    try:
        with open(yaml_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        # Extract the requested values
        model_config = config["model"]["model_config"]
        parallel_config = config["parallel_config"]
        runner_config = config["runner_config"]

        # Create a dictionary with the requested parameters
        extracted_params = {
            "num_heads": model_config["num_heads"],
            "hidden_size": model_config["hidden_size"],
            "head_dim": int(model_config["hidden_size"] / model_config["num_heads"]),
            "seq_length": model_config["seq_length"],
            "batch_size": runner_config["batch_size"],
            "vocab_size": model_config["vocab_size"],
            "num_layers": model_config["num_layers"],
            "data_parallel": parallel_config.get("data_parallel", 1),
            "model_parallel": parallel_config.get("model_parallel", 1),
            "context_parallel": parallel_config.get("context_parallel", 1),
            "pipeline_stage": parallel_config.get("pipeline_stage", 1),
            "micro_batch_num": parallel_config.get("micro_batch_num", 1)
        }

        return extracted_params

    except (yaml.YAMLError, FileNotFoundError, PermissionError) as e:
        logger.error(f"Error parsing Training YAML file: {e}")
        return None
