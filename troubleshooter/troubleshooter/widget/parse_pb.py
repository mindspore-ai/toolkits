# Copyright 2023 Huawei Technologies Co., Ltd
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
import csv
import json
import os

from .graph import mindinsight_anf_ir_pb2 as anf_ir_pb2
from .graph.graph import MSGraph, OptimizedGraph
from .graph.util import check_invalid_pb_file

TITLE = ('op_name',
         'op_full_name',
         'op_type',
         'precision_flag',
         'input or output',
         'index',
         'input from op or output to op',
         'shape',
         'dtype')


class PbParser:

    @staticmethod
    def parse_pb_file(file_path, mode='normal'):
        """
        Parse pb file and write content to `EventsData`.

        Args:
            file_path (str): The file path of pb file.

        Returns:
            TensorEvent, if load pb file and build graph success, will return tensor event, else return None.
        """

        model_proto = anf_ir_pb2.ModelProto()
        try:
            with open(file_path, 'rb') as fr:
                model_proto.ParseFromString(fr.read())

            graph = MSGraph() if mode == 'normal' else OptimizedGraph()
            graph.build_graph(model_proto.graph)
            print("Build graph success, file path: %s.", file_path)
            return graph
        except Exception:
            print("Warning: The given file is not a valid pb file, file path: %s.", file_path)
            return None


def get_all_node_detail(graph, scope, all_nodes_detail):
    """
    get all the nodes detail information recursively.

    Args:
        scope (str): The name of node.
        all_nodes_detail (list): all node info

    Returns:
        list, format is:
            item_object = [{'nodes': [<Node object>],
                   'scope_name': '',
                   'children': {<item_object>}}]
    """
    if scope and not graph.exist_node(scope):
        raise Exception(f"Node not in Graph : {scope}")

    nodes = graph.list_node_by_scope(scope=scope)
    if not isinstance(nodes, list):
        raise Exception(f"Node not in Graph : {scope}")
    if not nodes:
        return all_nodes_detail
    for node in nodes:
        name = node.get("name")
        sub_scope = node.get("subnode_count")
        if sub_scope == 0:
            all_nodes_detail.append(node)
            continue
        get_all_node_detail(graph, name, all_nodes_detail)
    return all_nodes_detail


def precision_tracker(abs_pb_filepath, precision_flags=('normal', 'raise', 'reduce'), **kwargs):
    """
    Parse the pd file, ouput the op nodes in csv format.
    """
    check_invalid_pb_file(abs_pb_filepath)
    if not isinstance(precision_flags, (list, tuple)):
        raise Exception(f'The type of arg precision_flags must be list or tuple, '
                        f'but got {type(precision_flags)}')
    abs_pb_filepath = os.path.abspath(abs_pb_filepath)
    output_path = kwargs.get('output_path')
    output_filename = kwargs.get('output_filename')
    if not output_path:
        output_path = os.path.dirname(abs_pb_filepath)
    if not output_filename:
        output_filename = os.path.basename(abs_pb_filepath)
    output_filename = output_filename.split('.')[0] + '.csv'

    graph = PbParser.parse_pb_file(abs_pb_filepath)
    all_nodes_detail = []
    get_all_node_detail(graph, '', all_nodes_detail)
    final_data = [TITLE]

    for node in all_nodes_detail:
        op_ull_name = node.get('name')
        op_name = op_ull_name.split('/')[-1]
        op_type = node.get('type')
        op_attr = node.get('attr', {})
        precision_flag = op_attr.get('precision_flag', 'normal')
        if precision_flag != 'normal':
            precision_flag = precision_flag.split(':')[-1]
            precision_flag = json.loads(precision_flag.strip())

        if precision_flag not in precision_flags:
            continue

        inputs = node.get('input', {})
        outputs = node.get('output', {})

        input_index = 0
        for input_full_name, info in inputs.items():
            input_name = input_full_name.split('/')[-1]
            shape = info.get('shape', '-')
            data_type = info.get('data_type', '-')
            final_data.append(
                (op_name, op_ull_name, op_type, precision_flag, 'input', input_index, input_name, shape, data_type))
            input_index += 1

        output_index = 0
        for output_full_name, info in outputs.items():
            output_name = output_full_name.split('/')[-1]
            shape = info.get('shape', '-')
            data_type = info.get('data_type', '-')
            final_data.append(
                (op_name, op_ull_name, op_type, precision_flag, 'output', output_index, output_name, shape, data_type))
            output_index += 1

    output_file = os.path.join(output_path, output_filename)
    flags = os.O_WRONLY | os.O_CREAT
    with os.fdopen(os.open(output_file, flags, 0o400), 'w', newline='') as fw:
        writer = csv.writer(fw)
        writer.writerows(final_data)

    print(f"文件分析完成，请查看结果文件：{output_file}")
