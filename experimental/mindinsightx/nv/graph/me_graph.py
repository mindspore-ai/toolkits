# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""Computational graph of MindSpore."""
import os.path
import onnx
import re

from google.protobuf import text_format, message

from mindinsightx.nv import constants
from mindinsightx.nv.graph.graph import Graph
from mindinsightx.nv.graph.graph import Node
from mindinsightx.nv.mapper.me_dump_mapper import MeDumpMapper
import mindinsightx.compare.proto.me.mindinsight_anf_ir_pb2 as me_ir_pb2
from mindinsightx.nv.mapper.mappings import mappings


class MeNode(Node):
    """Computational graph node of MindSpore."""

    @property
    def topo_id(self):
        """Get topological id."""
        return self.node_id


class MeGraph(Graph):
    """Computational graph of MindSpore."""

    NODE_TYPE_PARAMETER = 'Parameter'
    NODE_NAME_SUFFIX_MOVING_MEAN = 'moving_mean'
    NODE_NAME_SUFFIX_MOVING_VAR = 'moving_variance'

    def __init__(self):
        super().__init__()
        self._proto = None

    @property
    def graph_type(self):
        return constants.ME_GRAPH_TYPE

    @property
    def framework_version(self):
        """Get version of deep learning framework."""
        raise NotImplementedError

    @property
    def proto(self):
        """Get the raw proto object."""
        return self._proto

    def keep_shape(self, node_type):
        """Check if the output shape has to be kept in node mapping."""
        raise NotImplementedError

    def keep_shape_remove_batch(self, node_type):
        """Check if the output shape has to be keep with batch size removed in node mapping."""
        raise NotImplementedError

    def get_node_type_map(self, type_ver_code):
        """Get node type map."""
        raise NotImplementedError

    def create_dump_mapper(self, dump_dir):
        return MeDumpMapper(self, dump_dir)

    def _get_model_proto(self):
        return me_ir_pb2.ModelProto()

    def load(self, path):
        """
        Loads graph file.

        Args:
            path (str): Graph file path to be loaded.
        """
        _, ext = os.path.splitext(path)
        if ext == '.pb':
            self.real_path = os.path.realpath(path)
            self._proto = self._get_model_proto()

            try:
                with open(self.real_path, 'rb') as f:
                    self._proto.ParseFromString(f.read())
            except message.DecodeError:
                try:
                    with open(self.real_path, 'r') as f:
                        text_format.Merge(f.read(), self._proto)
                except Exception:
                    raise ValueError(f'Cannot load MindSpore graph file {self.real_path}.')

            id_node_map = dict()

            graph_def = self._proto.graph

            for node_def in graph_def.node:
                node = self._parse_proto_node(node_def)
                if not node.node_id or not node.name:
                    continue
                self.add_node(node)
                id_node_map[node.node_id] = node

            for param_def in graph_def.parameters:
                node = self._parse_proto_parameter(param_def)
                if not node.node_id or not node.name:
                    continue
                self.add_node(node)
                id_node_map[node.node_id] = node

            for node_def in graph_def.node:
                node = self.get_node_by_name(node_def.full_name)
                if node is None:
                    continue

                for input_def in node_def.input:
                    input_node = id_node_map.get(input_def.name, None)
                    if input_node is None:
                        continue
                    self.add_edge(input_node, node)
        elif ext == '.pbtxt':
            self.real_path = os.path.realpath(path)
            self._proto = onnx.ModelProto()
            try:
                with open(self.real_path, 'r') as f:
                    text_format.Parse(f.read(), self._proto)
            except Exception:
                raise ValueError(f'Cannot load MindSpore graph file {self.real_path}.')
            
            node_map = dict()

            graph_def = self._proto.graph
            id = 0

            for node_def in graph_def.node:
                id += 1
                op_type = node_def.op_type.replace("ge:", "")
                if op_type in mappings:
                    op_type = mappings[op_type]
                node = MeNode(name=node_def.name,
                            node_id=id,
                            node_type=op_type,
                            scope='Default')
                if not node.node_id or not node_def.name:
                    continue

                shapes = []
                for attr in node_def.attribute:
                    if attr.name == "output_desc_origin_shape:0":
                        shapes = list(attr.ints)
                node.shape = shapes
                self.add_node(node)
                node_map[node.name] = node

            for node_def in graph_def.node:
                node = self.get_node_by_name(node_def.name)
                if node is None:
                    continue

                for input_def in node_def.input:
                    input_node_name = input_def.split(':')[0]
                    input_node = node_map.get(input_node_name)
                    if input_node is None:
                        continue
                    self.add_edge(input_node, node)
        else:
            raise ValueError("File format not supported")

    def _parse_proto_node(self, node_def):
        """
        Parse `anf_ir_pb2.model_proto.graph.node_def`, and create a node.

        Args:
            node_def (anf_ir_pb2.model_proto.graph.node_def): See
                anf_ir_pb2.model_proto.graph.node_def.

        Returns:
            Node, a `Node` object.
        """
        if node_def.full_name:
            node_name = node_def.full_name
        else:
            # backward compatible
            node_name = node_def.name
        node = MeNode(name=node_name,
                      node_id=node_def.name,
                      node_type=node_def.op_type,
                      scope=node_def.scope)
        node.shape = self._parse_type_proto(node_def.output_type)
        return node

    def _parse_proto_parameter(self, param_def):
        """
        Parse anf_ir_pb2.model_proto.graph.parameter, and create a parameter node.

        Args:
            param_def (anf_ir_pb2.model_proto.graph.parameter): See
                anf_ir_pb2.model_proto.graph.parameter.

        Returns:
            Node, a `Node` object.
        """
        node = MeNode(name=param_def.name,
                      node_id=param_def.name,
                      node_type=self.NODE_TYPE_PARAMETER,
                      scope=self.NODE_TYPE_PARAMETER)
        return node

    @staticmethod
    def _parse_type_proto(type_proto):
        """
        Parse proto's `message TypeProto` to get shape information.

        Args:
            type_proto (anf_ir_pb2.TypeProto): See anf_ir_pb2.TypeProto.

        Returns:
            list, a list of shape.
        """
        shapes = []
        if type_proto.HasField('tensor_type'):
            tensor_type = type_proto.tensor_type
            tensor_shape_proto = tensor_type.shape
            for dim in tensor_shape_proto.dim:
                shapes.append(dim.size)
        return shapes

    def _get_node_type_for_label(self, node):
        """Get type of a member node for node labelling."""
        node_type = node.node_type
        if node_type == self.NODE_TYPE_PARAMETER:
            if node.name.endswith(self.NODE_NAME_SUFFIX_MOVING_MEAN):
                node_type = '_' + self.NODE_NAME_SUFFIX_MOVING_MEAN
            elif node.name.endswith(self.NODE_NAME_SUFFIX_MOVING_VAR):
                node_type = '_' + self.NODE_NAME_SUFFIX_MOVING_VAR
        return node_type
