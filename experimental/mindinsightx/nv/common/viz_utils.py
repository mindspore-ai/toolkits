# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Utilities for visualizations."""
import json

from google.protobuf.json_format import MessageToDict

from mindinsightx.nv.graph.me_graph import MeGraph
from mindinsightx.nv.comparator.graph_comparator import \
    DUMP_EQUAL, DUMP_EQUAL_SOME_MISSING, DUMP_MISSING, DUMP_NO_COMMON_INDEX


def save_me_as_json(comparator, graph_idx, filename, indent=4):
    """Save me graph and the associated comparison result to a json file."""
    graph = comparator.graphs[graph_idx]
    if not isinstance(graph, MeGraph):
        raise TypeError(f"comparator.graphs[{graph_idx}] is not a MeGraph.")
    if graph.proto is None:
        raise ValueError(f"The graph is not loaded from protobuf.")

    graph_dict = MessageToDict(graph.proto)["graph"]
    node_dicts = graph_dict["node"]
    for node in graph.nodes:
        labels = []
        node_pair = comparator.node_mapper.map_strict(graph_idx, node.internal_id)
        if node_pair is None:
            labels.append("unmapped")
        elif comparator.output_cmp_results is not None:
            cmp_result = comparator.output_cmp_results[graph_idx].get(node.internal_id, None)
            if cmp_result is None or cmp_result.value_cmp_result in (DUMP_MISSING, DUMP_NO_COMMON_INDEX):
                labels.append("output_missing")
            elif cmp_result.value_cmp_result not in (DUMP_EQUAL, DUMP_EQUAL_SOME_MISSING):
                labels.append("output_neq")
        if labels:
            for node_dict in node_dicts:
                if node_dict["name"] == node.node_id:
                    node_dict["nv_labels"] = labels
                    break

    data = {"message": "success", "data": graph_dict}
    with open(filename, "w") as fp:
        json.dump(data, fp, indent=indent)
