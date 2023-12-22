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
"""Example code."""

from mindinsightx.nv import constants
from mindinsightx.nv.graph.graph import Graph
from mindinsightx.nv.comparator.graph_comparator import GraphComparator
from mindinsightx.nv.mapper.node_mapper import NodeMapper


def comparator_with_dump_dir():

    """Test comparator with dump dir example."""

    graph0 = Graph.create(constants.ME_GRAPH_TYPE, '1.0.0-opt')
    graph0.load('me_graph.pb')
    graph1 = Graph.create(constants.TF_GRAPH_TYPE, '1.14')
    graph1.load('tf_graph.pb')

    comparator = GraphComparator(graph0, graph1, 'me_dump_dir', 'tf_dump_dir')
    comparator.compare()
    print('Saving as xlsx report...')
    comparator.save_as_xlsx('report.xlsx')


def comparator_without_dump_dir():

    """Comparator without dump dir example."""

    graph0 = Graph.create(constants.ME_GRAPH_TYPE, '1.0.0')
    graph0.load('ms_output_before_hwopt_0.pb')
    graph1 = Graph.create(constants.TF_GRAPH_TYPE, '1.14')
    graph1.load('tf_graph.pb')

    comparator = GraphComparator(graph0, graph1)
    comparator.compare()
    print('Saving as xlsx report...')
    comparator.save_as_xlsx('report.xlsx')


def node_mapper():
    """Node mapper example."""

    graph0 = Graph.create(constants.ME_GRAPH_TYPE, '1.0.0')
    graph0.load('me_graph.pb')
    graph1 = Graph.create(constants.TF_GRAPH_TYPE, '1.14')
    graph1.load('tf_graph.pb')

    mapper = NodeMapper(graph0, graph1)
    mapper.process()
    pair = mapper.map(0, graph0.nodes[0].internal_id)
    if pair:
        print(f'Mapped node: {pair[1].name} similarity: {pair.similarity}')
    top_k_pairs = mapper.top_k(0, graph0.nodes[0].internal_id, k=3)
    if top_k_pairs:
        print(f'Top-{len(top_k_pairs)} similar nodes: {[pair[1].name for pair in top_k_pairs]}')
