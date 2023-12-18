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
"""Computational graph and node."""

import re
import sys

from mindinsightx.nv import constants


class Node:
    """
    Computational graph node, a node represents an operator.

    Args:
        name (str): Node name.
        node_id (str): Node(operator) id.
        node_type (str): Node(operator) type.
        scope (str, optional): Scope.
    """
    def __init__(self, name, node_id, node_type, scope=None):
        self.name = name
        self.node_id = node_id
        self.node_type = node_type
        self.scope = scope
        self.shape = None
        self.label = None
        self.from_nodes = set()
        self.to_nodes = set()
        self.visited = False
        self.depth = -1
        self.homo_depth = -1
        self.internal_id = -1

        self.match_idx = -1
        self.match_sim = 0
        self.tmp_match_sim = 0
        self.tmp_matched_node = None
        self.partition = None
        self.footprints = None

    @property
    def topo_id(self):
        """Get topological id."""
        raise NotImplementedError

    @property
    def is_isolated(self):
        """Check if the node is not connected to another node."""
        return not self.from_nodes and not self.to_nodes

    def __hash__(self):
        """For acting as a dict key."""
        return id(self)

    def __eq__(self, other):
        """For acting as a dict key."""
        return self is other


class Graph:
    """Computational graph."""

    _NODE_TYPE_REMOVE_PATTERN = re.compile(r'\s|_|-|\.|[vV]\d+$')

    @staticmethod
    def create(graph_type=None, framework_version=None, type_ver_code=None):
        """
        Creates graph object.

        Args:
            graph_type (str, optional): Either constants.ME_GRAPH_TYPE or constants.TF_GRAPH_TYPE.
            framework_version (str, optional): The version of deep learning framework
            type_ver_code (str, optional): Graph type+version code, symbol like constants.ME_TYPE_VER_XXX
        Returns:
            Graph, The graph object.
        """
        if type_ver_code:
            if type_ver_code in [constants.ME_TYPE_VER_TRACE_CODE, constants.ME_TYPE_VER_TRACE_CODE_OPT]:
                from mindinsightx.nv.graph.me_trace_code_graph import \
                    MeTraceCodeGraph
                return MeTraceCodeGraph()

            splited = type_ver_code.split('-')
            if len(splited) >= 2:
                graph_type = splited[0]
                framework_version = '-'.join(splited[1:])
            else:
                graph_type = splited[0]
                framework_version = 'x'
        else:
            if not graph_type or not framework_version:
                raise ValueError(f"If type_ver_code is not provided "
                                 f"then graph_type and framework_version must be provided.")

        # get ms graph version
        if graph_type == constants.ME_GRAPH_TYPE:
            from mindinsightx.nv.graph.mex_graph import MeXGraph
            return MeXGraph(framework_version)

    def __init__(self):
        self.nodes = []
        self.name_nodes = dict()
        self.edges = []
        self._node_id_counter = 0
        self.real_path = ''

    @property
    def graph_type(self):
        """Get graph type."""
        raise NotImplementedError

    @property
    def framework_version(self):
        """Get version of deep learning framework."""
        raise NotImplementedError

    def keep_shape(self, node_type):
        """Check if the output shape has to be kept in node mapping."""
        raise NotImplementedError

    def keep_shape_remove_batch(self, node_type):
        """Check if the output shape has to be keep with batch size removed in node mapping."""
        raise NotImplementedError

    def get_node_type_map(self, type_ver_code):
        """Get node type map."""
        raise NotImplementedError

    @property
    def type_ver_code(self):
        """Get graph type-framework version code."""
        return self.graph_type+'-'+self.framework_version

    @property
    def node_names(self):
        """Get all node names."""
        return list(self.name_nodes.keys())

    def get_node_by_name(self, name):
        """Get node by name node name."""
        return self.name_nodes.get(name, None)

    def add_node(self, node):
        """Add a new node."""
        node.internal_id = self._node_id_counter
        self._node_id_counter += 1
        self.nodes.append(node)
        self.name_nodes[node.name] = node

    def add_edge(self, from_node, to_node):
        """Add a new edge."""
        from_node.to_nodes.add(to_node)
        to_node.from_nodes.add(from_node)
        self.edges.append((from_node, to_node))

    def remove_isolated_nodes(self):
        """Remove all isolated nodes."""
        count = len(self.nodes)
        i = 0
        while i < count:
            node = self.nodes[i]
            if node.is_isolated:
                self.nodes.pop(i)
                del self.name_nodes[node.name]
                count -= 1
            else:
                i += 1

    def create_dump_mapper(self, dump_dir):
        """Create a dump mapper."""
        del dump_dir

    def prepare_node_mapping(self, cmp_graph, ignore_shape=True):
        """
        Prepare for node mapping.

        Args:
            cmp_graph (Graph): Another graph to be aligned with.
            ignore_shape (bool): Ignore output shape.
        """
        if cmp_graph.type_ver_code == self.type_ver_code:
            # same graph type and framework, no mapping no specific shape policy
            node_type_map = None
            shape_policy = None
        else:
            node_type_map = self.get_node_type_map(cmp_graph.type_ver_code)
            shape_policy = self if node_type_map is None else cmp_graph

        for node in self.nodes:
            node.depth = -1
            node.homo_depth = -1
            node.label = self.label_node(node, ignore_shape, node_type_map, shape_policy)

        node_set = set(self.nodes)
        # _find_homo_depth()'s max. recursive depth is depends on the number of nodes
        limit_bak = sys.getrecursionlimit()
        sys.setrecursionlimit(1000 + len(self.nodes))

        for target_node in self.nodes:
            target_node.visited = False
            target_node.homo_depth = self._find_homo_depth(target_node, target_node.label, node_set)

        for target_node in self.nodes:
            target_node.visited = False
            target_node.depth = self._find_depth(target_node, node_set)

        sys.setrecursionlimit(limit_bak)

    def label_node(self, node, ignore_shape, node_type_map, shape_policy):
        """
        Label a member node.

        Args:
            node (Node): Member node to be labeled.
            ignore_shape (bool): Ignore output shape.
            node_type_map (dict[string, string]): Node type map.
            shape_policy (Graph): The graph that we should follow it's shape policy.

        Returns:
            str, The node label.
        """
        if shape_policy is None:
            # same graph type
            if ignore_shape:
                return self._to_label(node.node_type)
            return self._to_label(node.node_type, node.shape)

        node_type = self._get_node_type_for_label(node)
        if node_type_map is not None:
            node_type = node_type_map.get(node_type, node_type)

        if ignore_shape:
            return self._to_label(node_type)

        shape = None
        if node.shape:
            if shape_policy.keep_shape_remove_batch(node_type):
                shape = sorted(node.shape[1:])
            elif shape_policy.keep_shape(node_type):
                shape = sorted(node.shape)
        return self._to_label(node_type, shape)

    def _to_label(self, node_type, shape=None):
        """Generate node label from node type and shape."""
        node_type = self._NODE_TYPE_REMOVE_PATTERN.sub('', node_type.strip()).upper()
        if shape:
            shape_str = ','.join(map(str, shape))
            # return: e.g.'RELU|6,28,28'
            return f'{node_type}|{shape_str}'
        return node_type

    def _find_depth(self, node, node_set):
        """Find homogeneous depth."""
        if node.visited or node.depth > 0:
            return max(0, node.depth)

        node.visited = True
        node_set.add(node)
        max_depth = 0
        for from_node in node.from_nodes:
            depth = self._find_depth(from_node, node_set)
            if depth > max_depth:
                max_depth = depth
        return max_depth + 1

    def _find_homo_depth(self, node, label, visited_nodes=None):
        """Find homogeneous depth."""
        if node.visited:
            return 0
        if visited_nodes is None:  
            visited_nodes = set()
        node.visited = True
        visited_nodes.add(node)
        if node.label == label:
            if node.homo_depth > 0:
                return node.homo_depth

        max_depth = 0
        for from_node in node.from_nodes:
            if from_node not in visited_nodes:  
                depth = self._find_homo_depth(from_node, label, visited_nodes)
                if depth > max_depth:
                    max_depth = depth

        if node.label == label:
            return max_depth + 1

        return max_depth

    def _get_node_type_for_label(self, node):
        """Get type of a member node for node labelling."""
        return node.node_type
