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
"""stage"""

from typing import Union, Any
from dataclasses import dataclass

from toolkit.pipeline_balance.utils.error import _assert_sapp
import toolkit.pipeline_balance.utils.recompute as Recompute


class Stage:
    """Stage Class to describe a run from a log

    id_ (int): stage id of the run
    nb_stage_ (int): total number of stage present
    nb_layer_ (int): total number of layer present to distribute for this stage id
    nb_layer_rec_ (dict[Recompute.Type, int]): number of recomputed layer
        per recomputation type for this stage id
    memory_usage_ (int): memory usage of the run for this stage

    Properties:
    nb_layer_ == (nb_recompute+nb_select_rec+nb_norecompute)
    id_ < nb_stage_
    """

    id_: int
    nb_stage_: int
    nb_layer_: int
    nb_layer_rec_: dict[Recompute.TYPE, int]
    memory_usage_: int

    def __init__(self, sid: int, nb_stage: int, nb_layer: int,
                 nb_layer_rec: dict[Recompute.TYPE, int], memory_usage: int):
        self.id_ = sid
        self.nb_stage_ = nb_stage
        self.nb_layer_ = nb_layer
        self.nb_layer_rec_ = self.complete_nb_layer_rec_(nb_layer_rec)
        self.memory_usage_ = memory_usage
        _assert_sapp(nb_layer == Recompute.sums(nb_layer_rec),
                     "init stage, nb_layer == (nb_recompute+nb_norecompute)")
        _assert_sapp(sid < nb_stage, "init stage, id < nb_stage")

    def complete_nb_layer_rec_(self, nb_layer_rec: dict[Recompute.TYPE, int]):
        """Complete the number of layers recomputed from partial filling"""
        sum_layers = 0
        for r in Recompute.TYPE:
            if r is not Recompute.TYPE.NONE:
                if r not in nb_layer_rec:
                    nb_layer_rec[r] = 0
                else:
                    sum_layers += nb_layer_rec[r]

        if Recompute.TYPE.NONE not in nb_layer_rec:
            nb_layer_rec[Recompute.TYPE.NONE] = self.nb_layer_ - sum_layers

        return nb_layer_rec

    def same_config(self, other: 'Stage') -> bool:
        """
        Check if self and other have same configuration
        same number of total layers, number of total stages, recompute layers and no recompute
        layers
        """
        return (
            self.nb_layer_ == other.nb_layer_ and self.nb_stage_ == other.nb_stage_ and
            self.nb_layer_rec_ == other.nb_layer_rec_)

    def same_global_config(self, other: 'Stage') -> bool:
        """
        Check if self and other have same configuration
        same number of total layers and number of total stages
        """
        return self.nb_stage_ == other.nb_stage_

    def get_index_memory_var(self) -> list[int]:
        """
        Returns memory factors for parameter and
        activation for all recomputation types
        """
        diff = self.nb_stage_ - self.id_
        return [self.nb_layer_] + Recompute.to_list(
            {r: self.nb_layer_rec_[r] * diff for r in Recompute.TYPE})


def filter_stage_id(stages: list[Stage], stage_id: int) -> list[Stage]:
    """Filters all stages of stage_id in stages."""
    kept_stages = []
    for s in stages:
        if s.id_ == stage_id:
            kept_stages.append(s)
    return kept_stages


def process_offset(offset, pipeline_num):
    """process input offset"""
    rounds = 1
    if isinstance(offset, int) and offset == 0:
        offset = [0] * pipeline_num
    # if offset is list of lists (usually when pp=4)
    elif isinstance(offset, list) and any(isinstance(item, list) for item in offset):
        tmp_offset = []
        for item in offset:
            if isinstance(item, int) and item == 0:
                tmp_offset.append([0] * pipeline_num)
            elif not (isinstance(item, list) and len(item) == pipeline_num):
                raise ValueError(
                    f"Unsupported input format offset: {item},",
                    f"please check the length of your offset list and the pipeline number",
                )
            else:
                tmp_offset.append(item)
        offset = tmp_offset
        rounds = len(offset)
    elif not (isinstance(offset, list) and len(offset) == pipeline_num):
        raise ValueError(
            f"Unsupported input format offset: {offset},",
            "please check the length of your offset list and the pipeline number",
        )

    return offset, rounds


def process_rec_config(
    layer_per_stage: int, pipeline_num: int, offset: list[int], rec_config
):
    """process recomputation config into a dict"""
    if rec_config is None or offset is None:
        return None
    if isinstance(rec_config, bool):
        if rec_config:
            rec_config = [layer_per_stage] * pipeline_num
            rec_config = [recom + bias for recom, bias in (rec_config, offset)]
        else:
            rec_config = [0] * pipeline_num
        rec_config = [rec_config]
    elif isinstance(rec_config, list) and len(rec_config) == pipeline_num:
        # in order to be compatible with internal_from_yaml, change list into double list
        rec_config = [rec_config]
    else:
        raise ValueError(
            f"Unsupported input format recompute: {rec_config}, please check the length of list"
        )

    return rec_config


def instantiate_stage(stage_id, pipeline_num, nb_layer, layer_per_recompute, memory):
    """instantiate a stage"""
    stage = Stage(
        sid=stage_id,
        nb_stage=pipeline_num,
        nb_layer=nb_layer,
        nb_layer_rec={
            Recompute.TYPE.COMM: layer_per_recompute[Recompute.TYPE.COMM][0][stage_id],
            Recompute.TYPE.FULL: layer_per_recompute[Recompute.TYPE.FULL][0][stage_id],
            Recompute.TYPE.SLCT: layer_per_recompute[Recompute.TYPE.SLCT][0][stage_id],
            Recompute.TYPE.BOTH: layer_per_recompute[Recompute.TYPE.BOTH][0][stage_id],
        },
        memory_usage=memory,
    )
    return stage


@dataclass
class StagesBuilder:
    """build stages for compute_memory"""

    # pipeline config
    pipeline_num: int
    num_layer: int
    recompute_config: Any
    offset: Any
    # memory usage
    stage_ids: list
    mem_head_stage: Union[float, int]
    mem_tail_stage: Union[float, int]
    body_memories: list

    def __init__(
        self,
        pipeline_num,
        num_layer,
        recompute_config,
        offset,
        stage_ids,
        mem_head_stage,
        mem_tail_stage,
        body_memories,
    ):
        self.pipeline_num = pipeline_num
        self.num_layer = num_layer
        self.recompute_config = recompute_config
        self.offset = offset
        self.stage_ids = stage_ids
        self.mem_head_stage = mem_head_stage
        self.mem_tail_stage = mem_tail_stage
        self.body_memories = body_memories

    def build_stages(self):
        stages_a = []
        offset, rounds = process_offset(self.offset, self.pipeline_num)
        layer_per_stage = int(self.num_layer / self.pipeline_num)

        if rounds > 1:
            layer_per_recompute = []
            for i in range(rounds):
                offset_recompute = {}
                for rec in Recompute.YAML_NAME.values():
                    rec_list = self.recompute_config.get(rec)
                    if rec_list is None:
                        continue
                    offset_recompute[rec] = process_rec_config(
                        layer_per_stage, self.pipeline_num, offset[i], rec_list[i]
                    )
                offset_recompute[Recompute.OFFSET] = [offset[i]]
                layer_per_recompute.append(
                    Recompute.internal_from_yaml(
                        1,
                        self.pipeline_num,
                        offset_recompute,
                        [[layer_per_stage] * self.pipeline_num],
                    )
                )
        else:
            offset_recompute = {}
            for rec in Recompute.YAML_NAME.values():
                rec_list = self.recompute_config.get(rec)
                offset_recompute[rec] = process_rec_config(
                    layer_per_stage, self.pipeline_num, offset, rec_list
                )
            offset_recompute[Recompute.OFFSET] = [offset]
            layer_per_recompute = Recompute.internal_from_yaml(
                1,
                self.pipeline_num,
                offset_recompute,
                [[layer_per_stage] * self.pipeline_num],
            )

        if rounds > 1:
            for i in range(rounds):
                for idx, sg_id in enumerate(self.stage_ids[i]):
                    stages_a.append(
                        instantiate_stage(
                            sg_id,
                            self.pipeline_num,
                            layer_per_stage + offset[i][sg_id],
                            layer_per_recompute[i],
                            self.body_memories[i][idx],
                        )
                    )
            stages_a.append(
                instantiate_stage(
                    0,
                    self.pipeline_num,
                    layer_per_stage + offset[0][0],
                    layer_per_recompute[0],
                    self.mem_head_stage,
                )
            )
            stages_a.append(
                instantiate_stage(
                    self.pipeline_num - 1,
                    self.pipeline_num,
                    layer_per_stage + offset[0][self.pipeline_num - 1],
                    layer_per_recompute[0],
                    self.mem_tail_stage,
                )
            )
        else:
            for idx, sg_id in enumerate(self.stage_ids):
                stages_a.append(
                    instantiate_stage(
                        sg_id,
                        self.pipeline_num,
                        layer_per_stage + offset[sg_id],
                        layer_per_recompute,
                        self.body_memories[idx],
                    )
                )
            stages_a.append(
                instantiate_stage(
                    0,
                    self.pipeline_num,
                    layer_per_stage + offset[0],
                    layer_per_recompute,
                    self.mem_head_stage,
                )
            )
            stages_a.append(
                instantiate_stage(
                    self.pipeline_num - 1,
                    self.pipeline_num,
                    layer_per_stage + offset[self.pipeline_num - 1],
                    layer_per_recompute,
                    self.mem_tail_stage,
                )
            )

        return stages_a
