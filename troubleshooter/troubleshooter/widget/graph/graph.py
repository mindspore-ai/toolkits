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
import time
from collections import defaultdict

from .mindinsight_anf_ir_pb2 import DataType
from .node import Node
from .util import EdgeTypeEnum, NodeTypeEnum, PluginNameEnum, check_invalid_character


class Graph:
    """The `Graph` object is used to describe a graph file."""
    # Limit the size of a single attribute value per node to avoid storing too much data
    MAX_NODE_ATTRIBUTE_VALUE_BYTES = 1024

    # In the same scope, the number of children of the same type exceeds this threshold, and we will combine them.
    MIN_GROUP_NODE_COUNT = 5

    def __init__(self):
        # Used to cache all nodes, and the key is node name, value is `Node` object.
        self._normal_node_map = {}
        self._node_id_map_name = {}

        # The additional caching of Const and Parameter is to handle the Const
        # and Parameter nodes separately later.
        self._const_node_temp_cache = {}
        self._parameter_node_temp_cache = {}

        self._leaf_nodes = {}
        self._full_name_map_name = {}

    def build_graph(self, proto_data):
        """This method is used to build the graph."""
        print("Start to build graph")
        start_time = time.time()

        # Notice:
        # The following methods are interdependent and cannot be switched at will.
        self._parse_data(proto_data)
        self._add_variable_nodes(NodeTypeEnum.PARAMETER.value)
        self._build_aggregation_scope_nodes()
        self._process_independent_layout()
        self._build_name_scope_nodes()

        # Since const nodes are not aggregated, adding them at the end can save a lot of computation.
        self._add_variable_nodes(NodeTypeEnum.CONST.value)
        self._calc_subnode_count()
        self._leaf_nodes = self._get_leaf_nodes()
        self._full_name_map_name = self._get_leaf_node_full_name_map()

        precision = 3
        time_consuming = round((time.time() - start_time) * 1000, precision)
        print(f"Build graph end, all node count: {self.normal_node_count}, "
              f"const count: {len(self._const_node_temp_cache)}, "
              f"parameter count: {len(self._parameter_node_temp_cache)}, time-consuming: {time_consuming} ms.")

    @property
    def normal_node_count(self):
        """Get the normal node count."""
        return len(self._normal_node_map)

    def _add_variable_nodes(self, node_type):
        """
        We create the Const nodes or Parameter nodes in this method.

        Args:
            node_type (str): Decide which type of node to add.
                             Optional is `NodeTypeEnum.CONST.value` and `NodeTypeEnum.PARAMETER.value`.

        Note:
            This method relies on the presence of data in the const cache or parameter cache.
        """
        print(f"Start to add {node_type} nodes to each scope in graph.")
        node_map = {}
        for node in self._normal_node_map.values():
            for src_name, input_attr in dict(node.inputs).items():

                if node_type == NodeTypeEnum.CONST.value and not self._const_node_temp_cache.get(src_name):
                    continue

                if node_type == NodeTypeEnum.PARAMETER.value and not self._parameter_node_temp_cache.get(src_name):
                    continue

                variable_name = Node.create_node_name(scope=node.scope, base_name=src_name)
                if node_map.get(variable_name):
                    # There is no need to create the node repeatedly
                    variable_node = node_map.get(variable_name)
                else:
                    cache_node = self._get_normal_node(node_name=src_name)
                    variable_node = Node(name=variable_name, node_id=variable_name)
                    Node.copy_node_without_input_output(cache_node, variable_node)
                    variable_node.scope = node.scope

                variable_node.add_outputs(dst_name=node.name, output_attr=dict(input_attr))
                node_map.update({variable_name: variable_node})

                node.delete_inputs(src_name)
                node.add_inputs(variable_name, dict(input_attr))

        for node in node_map.values():
            self._cache_node(node)

        # Remove nodes that are not used in the cache.
        if node_type == NodeTypeEnum.CONST.value:
            unused_names = set(self._const_node_temp_cache) - set(node_map)
        elif node_type == NodeTypeEnum.PARAMETER.value:
            unused_names = set(self._parameter_node_temp_cache) - set(node_map)
        else:
            raise Exception("The node type should be const or parameter.")

        self._delete_nodes_of_cache(unused_names)

    def _delete_nodes_of_cache(self, node_names):
        """Delete node from cache."""
        print("These nodes will be removed from the cache, node names: %s.", str(node_names))
        for name in node_names:

            if self._parameter_node_temp_cache.get(name):
                self._parameter_node_temp_cache.pop(name)
            if self._const_node_temp_cache.get(name):
                self._const_node_temp_cache.pop(name)

            node = self._get_normal_node(node_name=name)
            self._normal_node_map.pop(name)
            self._node_id_map_name.pop(node.node_id)

    def _build_aggregation_scope_nodes(self):
        """
        Under the same scope, the number of nodes of the same type will be aggregated after exceeding the set threshold.

        Note:
            The threshold value refers to the `MIN_GROUP_NODE_COUNT`.
        """
        print("Start to build aggregation scope nodes.")
        group_node_map, filtered_group_names = self._find_group_nodes()

        # create merge scope nodes
        aggregation_scope_node_map = {}
        for i, group_name in enumerate(filtered_group_names):
            slash_index = group_name.rfind('/')
            if slash_index != -1:
                scope, op_type = group_name[:slash_index], group_name[slash_index + 1:]
            else:
                scope, op_type = '', group_name

            count = len(group_node_map.get(group_name))
            aggregation_node_name = Node.create_node_name(scope=scope, base_name=f'{op_type}[{count}]_{i}')
            aggregation_scope_node = Node(name=aggregation_node_name, node_id=aggregation_node_name)
            aggregation_scope_node.subnode_count = count
            aggregation_scope_node.scope = scope
            aggregation_scope_node.type = NodeTypeEnum.AGGREGATION_SCOPE.value

            # Update the name and scope of all children nodes
            for node in group_node_map[group_name]:
                base_name = node.name.split('/')[-1]
                new_name = Node.create_node_name(scope=aggregation_node_name, base_name=base_name)
                node.scope = aggregation_node_name

                # Since the name scope has not been created, there is no need to update the parent node.
                self._update_node_name_of_cache(node, new_name, update_parent=False)

            # Cache this node
            self._cache_node(aggregation_scope_node)
            aggregation_scope_node_map.update({group_name: aggregation_scope_node})

        # Adds the input and output of all direct child nodes to the current node.
        for group_name, node in aggregation_scope_node_map.items():
            self._inherit_input_output_from_subnode(node, group_node_map[group_name])

    def _find_group_nodes(self):
        """
        Find nodes that can be grouped into a group.

        For direct child nodes in a scope, we divide them into multiple groups by node type.
        However, we will exclude several types of child nodes,
        because these types of nodes are not operational nodes.
        """
        exclude_types = {
            NodeTypeEnum.CONST.value,
            NodeTypeEnum.NAME_SCOPE.value,
        }

        group_node_map = defaultdict(list)
        for node in self._normal_node_map.values():
            if node.type in exclude_types:
                continue
            group_name = Node.create_node_name(scope=node.scope, base_name=node.type)
            group_node_map[group_name].append(node)

        # filter can group scope.
        filtered_group_names = []
        for name, nodes in group_node_map.items():
            if len(nodes) < self.MIN_GROUP_NODE_COUNT:
                continue
            filtered_group_names.append(name)

        return group_node_map, filtered_group_names

    def _process_independent_layout(self):
        """Handle separate layout nodes."""
        independent_layout_node_map = {}
        for node in self._normal_node_map.values():
            base_name = node.name.split('/')[-1]
            if node.type == NodeTypeEnum.AGGREGATION_SCOPE.value and NodeTypeEnum.PARAMETER.value in base_name:
                independent_layout_node_map[node.name] = node

        # Find all sub nodes
        subnode_map = defaultdict(list)
        for node in self._normal_node_map.values():
            if independent_layout_node_map.get(node.scope):
                subnode_map[node.scope].append(node)

        # Notice:
        # The following processing is only done for the parameter node, other types of nodes are not processed.
        # Later, when you need to extend to other nodes, the code needs to be adjusted.
        for scope_node in independent_layout_node_map.values():
            scope_node.independent_layout = True

            method = 'outputs'
            for target_name, target_attr in dict(getattr(scope_node, method)).items():
                proxy_attr = dict(edge_type=target_attr['edge_type'])

                target_node = self._get_normal_node(node_name=target_name)
                getattr(target_node, 'add_proxy_inputs')(scope_node.name, proxy_attr)

                # Note:
                # If the source node and the destination node are not in the same scope,
                # the proxy node is presented as scope in order to simplify the flow of the display data.
                # For example, the data flow is parameter[5]_1 -> add[5]_1/add1
                # we create a scope proxy node(add[5]_1) for parameter[5]_1,
                # so there is a proxy data flow parameter[5]_1 -> add[5]_1 instead of parameter[5]_1 -> add[5]_1/add1.
                if target_node.scope == scope_node.scope:
                    getattr(scope_node, f'add_proxy_{method}')(target_name, proxy_attr)
                else:
                    target_scope_node = self._get_normal_node(node_name=target_node.scope)
                    getattr(scope_node, f'add_proxy_{method}')(target_node.scope, proxy_attr)
                    getattr(target_scope_node, 'add_proxy_inputs')(scope_node.name, proxy_attr)

            for subnode in subnode_map[scope_node.name]:
                subnode.independent_layout = True
                for target_name, target_attr in dict(getattr(subnode, method)).items():
                    proxy_attr = dict(edge_type=target_attr['edge_type'])
                    target_node = self._get_normal_node(node_name=target_name)
                    if target_node.scope == scope_node.scope:
                        getattr(subnode, f'add_proxy_{method}')(target_name, proxy_attr)
                    else:
                        getattr(subnode, f'add_proxy_{method}')(target_node.scope, proxy_attr)

                    input_attr = getattr(target_node, 'inputs')[subnode.name]
                    input_attr['independent_layout'] = True
                    target_node.add_inputs(subnode.name, input_attr)

    def _build_name_scope_nodes(self):
        """
        Build name scope node by every node name.

        We create the name scope node by the slash('/') in the node name.
        For example, if a node name is "Default/add", we generate a scope named 'Default' based on slash('/') and
        create a name scope node named 'Default'.
        """
        print("Start to build name scope nodes.")
        scope_node_map = {}
        for name, node in self._normal_node_map.items():
            index = name.find('/')
            pre_index = None
            while index > 0:
                scope = name[:index]
                scope_node = scope_node_map.get(scope)
                if scope_node is None:
                    if self._is_node_exist(node_name=scope):
                        exist_node = self._get_normal_node(node_name=scope)
                        if exist_node.type == NodeTypeEnum.AGGREGATION_SCOPE.value:
                            # This scope is aggregation scope, so we don't have to do anything.
                            pre_index = index
                            index = name.find('/', pre_index + 1)
                            continue

                        # We find a node name that conflicts with the current scope and rename the node
                        self._update_conflict_node(conflict_name=scope)

                    # We create a node for current scope.
                    scope_node = Node(scope, node_id=scope)
                    scope_node.type = NodeTypeEnum.NAME_SCOPE.value
                    scope_node.scope = '' if pre_index is None else name[:pre_index]
                    scope_node_map.update({scope_node.name: scope_node})

                # Inherit input and output from sub nodes.
                self._inherit_input_output_from_subnode(scope_node, subnode_list=[node])

                pre_index = index
                index = name.find('/', pre_index + 1)

        # Cache all the scope node to normal node dict
        for node in scope_node_map.values():
            self._cache_node(node)

    def _get_normal_node(self, node_id=None, node_name=None):
        """Query node by node id or node name."""
        if node_id is not None:
            name = self._node_id_map_name.get(node_id)
            node = self._normal_node_map.get(name)
            return node

        if node_name is not None:
            return self._normal_node_map.get(node_name)

        raise Exception('Method requires an argument that is not None.')

    def _is_node_exist(self, node_id=None, node_name=None):
        """Check node is exist."""
        if node_id is not None:
            return bool(self._node_id_map_name.get(node_id))

        if node_name is not None:
            return bool(self._normal_node_map.get(node_name))

        raise Exception('Method requires an argument that is not None.')

    def _inherit_input_output_from_subnode(self, parent_node, subnode_list, filtered_type=None):
        """
        Adds the input and output of all direct child nodes to the current node.

        Args:
            parent_node (Node): The nodes that inherit the input and output of the child nodes.
            subnode_list (list[Node]): A list of child nodes that are inherited from the input and output.
            filtered_type (set(str)): Filter some input and output that do not require inheritance
                                      based on the node type. Default is filter const node.

        Note:
            - Only the inputs and outputs of the external scope are inherited.
            - Before add_const_node method, if the input is a const,
              the scope of the const node is not startswith the name of parent node.
              So in this scenario, we need to filter the const nodes.
        """
        filtered_type = {NodeTypeEnum.CONST.value} if filtered_type is None else filtered_type
        for method in ['inputs', 'outputs', 'proxy_inputs', 'proxy_outputs']:
            for node in subnode_list:
                for item_name, item_attr in getattr(node, method).items():
                    target_node = self._get_normal_node(node_name=item_name)
                    if item_name.startswith(f'{parent_node.name}/'):
                        # Own scope, ignore
                        continue

                    if target_node.type in filtered_type:
                        continue

                    getattr(parent_node, f'add_{method}')(item_name, item_attr)

    def _update_conflict_node(self, conflict_name):
        conflict_node = self._get_normal_node(node_name=conflict_name)
        base_name = conflict_name.split('/')[-1]
        new_name = Node.create_node_name(scope=conflict_node.scope, base_name=f'({base_name})')
        self._update_node_name_of_cache(conflict_node, new_name, update_parent=True)

    def _calc_subnode_count(self):
        """Calc all the direct sub node count."""
        subnode_count_map = defaultdict(int)
        for node in self._normal_node_map.values():
            if not node.scope:
                continue

            if not self._is_node_exist(node_name=node.scope):
                print(f"Warning: Can not find a scope node by the given name({node.scope}), "
                      f"the name scope nodes may not have been created.")
                continue
            subnode_count_map[node.scope] = subnode_count_map[node.scope] + 1

        for name, count in subnode_count_map.items():
            node = self._get_normal_node(node_name=name)
            node.subnode_count = count

    def _get_leaf_nodes(self):
        """
        Get all leaf nodes, including normal leaf nodes, const nodes and param nodes.
        """
        leaf_nodes = {}
        for node_name, node in self._normal_node_map.items():
            # update full name
            if not node.full_name:
                node.full_name = node.name
            if not node.type or node.type.endswith('_scope'):
                continue
            leaf_nodes[node_name] = node

        return leaf_nodes

    def _get_leaf_node_full_name_map(self):
        """Get node by debugger name."""
        full_name_map = {}
        for name, node in self._leaf_nodes.items():
            if not node.full_name:
                print("Warning: Node %s does not have full name.", name)
                continue
            full_name_map[node.full_name] = name

        return full_name_map

    def _cache_node(self, node):
        """Store the node in the cache."""
        # Notice:
        # The additional caching of Const and Parameter is to handle the Const and Parameter nodes separately later.
        if node.type == NodeTypeEnum.CONST.value:
            self._const_node_temp_cache.update({node.name: node})
        if node.type == NodeTypeEnum.PARAMETER.value:
            self._parameter_node_temp_cache.update({node.name: node})

        self._normal_node_map.update({node.name: node})
        self._node_id_map_name.update({node.node_id: node.name})

    def _parse_data(self, proto_data):
        """
        This method will parse the data and create basic nodes to store in the cache.

        The graph is then built based on the cache.
        """
        raise NotImplementedError("Before you can build a graph, you need to parse the data.")

    def list_node_by_scope(self, scope=None):
        """
        List nodes by the scope of nodes. The scope of a node is the same as its parent node name.

        Args:
            scope (str): A scope of nodes.

        Returns:
            list[dict], a list object contain `Node` object.
        """
        scope = "" if scope is None else scope
        nodes = []
        for node in self._normal_node_map.values():
            if node.scope == scope:
                nodes.append(node.to_dict())
        return nodes

    def exist_node(self, name):
        """
        Check node exist in graph.

        Args:
            name (str): The node name.

        Returns:
            bool, if node exists, will return True.

        """
        if name is None:
            return False
        return self._is_node_exist(node_name=name)

    def _update_node_name_of_cache(self, node, new_name, update_parent=False):
        """
        Update a node name which is stored in cache.

        Args:
            node (Node): The node that will be renamed.
            new_name (str): The new name.
            update_parent (bool): Determines whether the input and output of the parent node need to be updated.
        """
        print('Update node name of cache, node(%s), new name is %s.', str(node), new_name)
        origin_name = node.name
        node.name = new_name
        node.scope = new_name.rsplit('/', 1)[0]

        # Find all nodes that need to modify the input and input
        update_node_map = {}
        for method in ['inputs', 'outputs', 'proxy_inputs', 'proxy_outputs']:
            for target_name in getattr(node, method):
                target_node = self._get_normal_node(node_name=target_name)
                if target_node is None:
                    message = f"Node should not be None, name: {target_name}, {method}: {list(getattr(node, method))}."
                    print('Error:', message)
                    continue

                update_node_map.update({target_name: target_node})

                if not update_parent:
                    continue

                slash_index = target_name.find('/')
                while slash_index != -1:
                    scope_name = target_name[:slash_index]
                    slash_index = target_name.find('/', slash_index + 1)

                    if update_node_map.get(scope_name):
                        continue

                    scope_node = self._get_normal_node(node_name=scope_name)
                    if scope_node is None:
                        message = f"Can not find the scope node by scope name({scope_name}), " \
                                  f"may be this scope node has not been built."
                        print(message)
                        continue

                    update_node_map.update({scope_name: scope_node})

        # Update the input and output of the nodes
        for target_node in update_node_map.values():
            for method in ['inputs', 'outputs', 'proxy_inputs', 'proxy_outputs']:
                attr_temp = getattr(target_node, method).get(origin_name)
                if attr_temp is None:
                    # This method does not have this node, so it is skipped
                    continue

                # Delete the old attribute and update new name to source node or destination node.
                getattr(target_node, f'delete_{method}')(origin_name)
                getattr(target_node, f'add_{method}')(new_name, attr_temp)

        # Delete the origin node in cache.
        self._delete_nodes_of_cache(node_names=[origin_name])
        self._cache_node(node)


class MSGraph(Graph):
    """The object describes the MindSpore graph, and it is defined in the anf_ir proto file."""

    def _parse_data(self, proto_data):
        """
        The proto data is parsed and all nodes are stored in the specified structure.

        Args:
            proto_data (anf_ir_pb2.GraphProto): Refer to anf_ir_pb2.GraphProto object.
        """
        self._parse_op_nodes(proto_data.node)
        self._parse_parameters(proto_data.parameters)
        self._parse_consts(proto_data.const_vals)

        self._update_input_after_create_node()
        self._update_output_after_create_node()

    def _parse_op_nodes(self, node_protos):
        """
        Parse `anf_ir_pb2.NodeProto` object, and create a normal node.

        Args:
            node_protos (list[anf_ir_pb2.NodeProto]): Refer to anf_ir_pb2.NodeProto.
        """
        print("Start to parse nodes from proto.")
        for topological_index, node_proto in enumerate(node_protos):
            if not node_proto.name:
                print("warning: Finding a node with an empty name will not save it.")
                continue

            self._parse_op_node(topological_index, node_proto)

    def _parse_op_node(self, topological_index, node_proto):
        """Parse `anf_ir_pb2.NodeProto` object, and create a normal node."""
        name = node_proto.name.split('/')[-1]
        node_id = name.split('op')[-1]
        name = f'{node_proto.op_type}-op{node_id}'
        node_name = Node.create_node_name(node_proto.scope, name)

        if node_proto.full_name and node_proto.op_type != NodeTypeEnum.LOAD.value:
            node_name = node_proto.full_name

        if node_proto.full_name and any(
                node_proto.full_name.lower().endswith(f'[:{plugin.value.lower()}]') for plugin in PluginNameEnum):
            node_name = Node.create_node_name(node_proto.scope, f'{node_proto.op_type}-op{node_proto.name}')

        # The Graphviz plug-in that the UI USES can't handle these special characters.
        check_invalid_character(node_name)

        node = Node(name=node_name, node_id=node_id, full_name=node_proto.full_name,
                    topological_index=topological_index)
        node.type = node_proto.op_type
        # if getattr(node_proto, 'source_address', None):
        #     node.stack = DebuggerSource.build_stack_from_source_address(node_proto.source_address)
        self._parse_attributes(node_proto.attribute, node)
        self._parse_inputs(node_proto.input, node)

        node.output_i = node_proto.output_i
        node.scope = node_proto.scope
        node.output_shape = self._get_shape_by_parse_type_proto(node_proto.output_type)
        node.output_nums = len(node.output_shape)
        node.output_data_type = self._get_data_type_by_parse_type_proto(node_proto.output_type, node)

        self._cache_node(node)

    def _parse_inputs(self, input_protos, node):
        """
        Parse `anf_ir_pb2.InputProto` object.

        Args:
            input_protos (list[anf_ir_pb2.InputProto]): Refer to `anf_ir_pb2.InputProto` object.
            node (Node): Refer to `Node` object, it is used to log message and update input.
        """
        for input_proto in input_protos:
            if not input_proto.name:
                print("Warning: The name in input proto of node(%s) is empty, will ignore.", node.name)
                continue

            if "/" in input_proto.name:
                input_proto.name = input_proto.name.split("op")[-1]

            edge_type = EdgeTypeEnum.DATA.value if not input_proto.type else EdgeTypeEnum.CONTROL.value

            # Notice:
            # 1. The name in the input proto is the node id of the Node object.
            # 2. In the current step, the shape of source node cannot be obtained,
            #    so it is set to empty list by default, and the next step will update it.
            # 3. Same with scope, set the default value first.
            input_attr = {
                "shape": [],
                "edge_type": edge_type,
                "independent_layout": False,
                'data_type': ''
            }

            node.add_inputs(src_name=input_proto.name, input_attr=input_attr)

    def _parse_attributes(self, attributes, node):
        """
        Parse `anf_ir_pb2.AttributeProto` object., and Filters large attribute values.

        Args:
            attributes (list[anf_ir_pb2.AttributeProto]): Refer to `anf_ir_pb2.AttributeProto` object.
            node (Node): Refer to `Node` object, it is used to log message and update attr.
        """
        for attr in attributes:
            if attr.value.ByteSize() > self.MAX_NODE_ATTRIBUTE_VALUE_BYTES:
                message = f"The attribute value of node({node.name}) " \
                          f"is over {self.MAX_NODE_ATTRIBUTE_VALUE_BYTES} Bytes, will ignore."
                print('Warning: ', message)
                continue
            if attr.name in ('input_is_dynamic_shape', 'output_is_dynamic_shape') and not \
                    node.is_dynamic_shape_node and attr.value.bool_val:
                node.is_dynamic_shape_node = True
            node.add_attr({attr.name: str(attr.value)})

    def _parse_parameters(self, parameter_protos):
        """
        Parse `anf_ir_pb2.ParameterProto` object, and create a parameter node.

        Args:
            parameter_protos (list[anf_ir_pb2.ParameterProto]): Refer to anf_ir_pb2.ParameterProto.
        """
        print("Start to parse parameters from proto.")
        for parameter in parameter_protos:
            if not parameter.name:
                print("Warning: Finding a parameter with an empty name will not save it.")
                continue
            check_invalid_character(parameter.name)
            node = Node(name=parameter.name, node_id=parameter.name)
            node.type = NodeTypeEnum.PARAMETER.value
            node.output_shape = self._get_shape_by_parse_type_proto(parameter.type)
            node.output_nums = len(node.output_shape)
            node.output_data_type = self._get_data_type_by_parse_type_proto(parameter.type, node)
            attr = dict(
                type=self._get_data_type_by_parse_type_proto(parameter.type, node),
                shape=str(self._get_shape_by_parse_type_proto(parameter.type))
            )
            node.add_attr(attr)

            self._cache_node(node)

    def _get_shape_by_parse_type_proto(self, type_proto):
        """
        Parse proto's `message TypeProto` to get shape information.

        Args:
            type_proto (anf_ir_pb2.TypeProto): Refer to anf_ir_pb2.TypeProto.

        Returns:
            list, a list of shape.
        """
        shapes = []
        if type_proto.HasField('data_type'):
            if type_proto.data_type != DataType.DT_TENSOR and \
                    type_proto.data_type != DataType.DT_TUPLE:
                # Append an empty list as a placeholder
                # for the convenience of output number calculation.
                shapes.append([])
                return shapes
        if type_proto.HasField('tensor_type'):
            tensor_type = type_proto.tensor_type
            tensor_shape_proto = tensor_type.shape
            shape = [dim.size for dim in tensor_shape_proto.dim]
            shapes.append(shape)
        if type_proto.HasField('sequence_type'):
            for elem_type in type_proto.sequence_type.elem_types:
                shapes.extend(self._get_shape_by_parse_type_proto(elem_type))
        return shapes

    def _get_data_type_by_parse_type_proto(self, type_proto, node):
        """
        Get data type by parse type proto object.

        The name of the DataType, refer to `anf_ir_pb2.DataType` object.
        If data type is tensor or tuple, the data name we return is `data_type[element_type, element_type]`.

        Args:
            type_proto (anf_ir_pb2.TypeProto): Refer to anf_ir_pb2.TypeProto.

        Returns:
            str, the data type.

        """
        data_type_name = self._get_data_type_name_by_value(type_proto, type_proto.data_type, field_name='data_type')
        if type_proto.data_type == DataType.DT_TENSOR:
            tensor_type_proto = type_proto.tensor_type
            value = type_proto.tensor_type.elem_type
            elem_type_name = self._get_data_type_name_by_value(tensor_type_proto, value, field_name='elem_type')
            node.elem_types.append(elem_type_name)
            return f'{data_type_name}[{elem_type_name}]'

        if type_proto.data_type == DataType.DT_TUPLE:
            data_types = []
            for elem_type in type_proto.sequence_type.elem_types:
                data_types.append(self._get_data_type_by_parse_type_proto(elem_type, node))
            return f'{data_type_name}{str(data_types)}'

        node.elem_types.append(data_type_name)

        return data_type_name

    def _parse_consts(self, consts):
        """
        Parse `anf_ir_pb2.NameValueProto` object, and create a const node.

        Args:
            consts (list[anf_ir_pb2.NameValueProto]): Refer to `anf_ir_pb2.NameValueProto` object.
        """
        print("Start to parse consts from proto.")
        for const in consts:
            if not const.key:
                print("Warning: Finding a const with an empty key will not save it.")
                continue
            check_invalid_character(const.key)
            node = Node(name=const.key, node_id=const.key, full_name=const.full_name)
            node.type = NodeTypeEnum.CONST.value
            if const.value.ByteSize() > self.MAX_NODE_ATTRIBUTE_VALUE_BYTES:
                node.add_attr({const.key: 'dtype: ' + DataType.Name(const.value.dtype)})
            else:
                node.add_attr({const.key: str(const.value)})

            if const.value.dtype == DataType.DT_TENSOR:
                shape = list(const.value.tensor_val.dims)
                node.output_shape.append(shape)
                if const.value.tensor_val.HasField('data_type'):
                    node.elem_types.append(DataType.Name(const.value.tensor_val.data_type))
            else:
                node.elem_types.append(DataType.Name(const.value.dtype))
                # dim is zero
                node.output_shape.append([])

            node.output_nums = len(node.output_shape)

            self._cache_node(node)

    def _update_input_after_create_node(self):
        """Update the input of node after create node."""
        for node in self._normal_node_map.values():
            for src_node_id, input_attr in dict(node.inputs).items():
                node.delete_inputs(src_node_id)
                if not self._is_node_exist(node_id=src_node_id):
                    message = f"The input node could not be found by node id({src_node_id}) " \
                              f"while updating the input of the node({node})"
                    print('Warning: ', message)

                    continue

                src_node = self._get_normal_node(node_id=src_node_id)
                input_attr['shape'] = src_node.output_shape
                input_attr['data_type'] = src_node.output_data_type
                node.add_inputs(src_name=src_node.name, input_attr=input_attr)

    def _update_output_after_create_node(self):
        """Update the output of node after create node."""
        # Constants and parameter should not exist for input and output.
        filtered_node = {NodeTypeEnum.CONST.value, NodeTypeEnum.PARAMETER.value}
        for node in self._normal_node_map.values():
            for src_name, input_attr in node.inputs.items():
                src_node = self._get_normal_node(node_name=src_name)
                if src_node.type in filtered_node:
                    continue

                src_node.add_outputs(node.name, input_attr)

    @staticmethod
    def _get_data_type_name_by_value(data_type, value, field_name='data_type'):
        """Get the data type name by the enum value, data_type refer to `DataType` object."""
        return data_type.DESCRIPTOR.fields_by_name[field_name].enum_type.values_by_number[value].name


class OptimizedGraph(MSGraph):
    """The object describes the MindSpore graph, and it is defined in the anf_ir proto file."""

    MIN_GROUP_NODE_COUNT = 10

    def __init__(self):
        super().__init__()
        self._load_node_temp_cache = {}

    def _inherit_input_output_from_subnode(self, parent_node, subnode_list, filtered_type=None):
        """
        Adds the input and output of all direct child nodes to the current node.

        Args:
            parent_node (Node): The nodes that inherit the input and output of the child nodes.
            subnode_list (list[Node]): A list of child nodes that are inherited from the input and output.
            filtered_type (set(str)): Filter some input and output that do not require inheritance
                                      based on the node type. Default is filter const node.

        Note:
            - Only the inputs and outputs of the external scope are inherited.
            - Before add_const_node method, if the input is a const,
              the scope of the const node is not startswith the name of parent node.
              So in this scenario, we need to filter the const nodes.
        """
        filtered_type = {NodeTypeEnum.CONST.value} if filtered_type is None else filtered_type
        for method in ['inputs', 'outputs', 'proxy_inputs', 'proxy_outputs']:
            for node in subnode_list:
                for item_name, item_attr in getattr(node, method).items():
                    target_node = self._get_normal_node(node_name=item_name)
                    if target_node is None:
                        print("Warning: inherit %s from subnode, target node (%s) is None", method, item_name)
                        continue

                    if item_name.startswith(f'{parent_node.name}/'):
                        continue

                    if target_node.type in filtered_type:
                        continue

                    getattr(parent_node, f'add_{method}')(item_name, item_attr)

    def _cache_node(self, node):
        """Store the node in the cache."""
        # Notice:
        # The additional caching is used to handle the Const, Parameter and LOAD nodes separately later.
        super()._cache_node(node)
        if node.type == NodeTypeEnum.LOAD.value:
            self._load_node_temp_cache.update({node.name: node})

    def _delete_nodes_of_cache(self, node_names):
        """Delete node from cache."""
        print("These nodes will be removed from the cache, node names: %s.", node_names)
        for name in node_names:

            if self._parameter_node_temp_cache.get(name):
                self._parameter_node_temp_cache.pop(name)
            if self._const_node_temp_cache.get(name):
                self._const_node_temp_cache.pop(name)
            if self._load_node_temp_cache.get(name):
                self._load_node_temp_cache.pop(name)

            node = self._get_normal_node(node_name=name)
            self._normal_node_map.pop(name)
            self._node_id_map_name.pop(node.node_id)

    def _parse_data(self, proto_data):
        """
        The proto data is parsed and all nodes are stored in the specified structure.

        Args:
            proto_data (anf_ir_pb2.GraphProto): Refer to anf_ir_pb2.GraphProto object.
        """
        print("Start to parse graph proto data.")

        self._parse_op_nodes(proto_data.node)
        self._parse_parameters(proto_data.parameters)
        self._parse_consts(proto_data.const_vals)

        self._update_input_after_create_node()
        self._update_output_after_create_node()

        self._delete_non_computational_ops()
        self._clean_no_input_output_node()

        self._extract_node_by_single_node_in_scope()
        print("Parse proto data end, normal node count(only contain op node, "
              "parameter, const): %s.", self.normal_node_count)

    def _parse_op_nodes(self, node_protos):
        """
        Parse `anf_ir_pb2.NodeProto` object, and create a normal node.

        Args:
            node_protos (list[anf_ir_pb2.NodeProto]): Refer to anf_ir_pb2.NodeProto.
        """
        print("Start to parse nodes from proto.")
        for topological_index, node_proto in enumerate(node_protos):
            if not node_proto.name:
                print("Warning: Finding a node with an empty name will not save it.")
                continue

            if node_proto.full_name.startswith("Gradients") or "optimizer" in node_proto.full_name \
                    or "opt" in node_proto.instance_name:
                continue

            self._parse_op_node(topological_index, node_proto)

    def _update_input_after_create_node(self):
        """Update the input of node after create node."""
        for node in self._normal_node_map.values():
            for src_node_id, input_attr in dict(node.inputs).items():
                node.delete_inputs(src_node_id)
                if not self._is_node_exist(node_id=src_node_id):
                    continue

                src_node = self._get_normal_node(node_id=src_node_id)
                input_attr['shape'] = src_node.output_shape
                input_attr['data_type'] = src_node.output_data_type
                node.add_inputs(src_name=src_node.name, input_attr=input_attr)

        nodes = self._list_nodes_without_parameter_const()
        for node in nodes:
            for src_node_name, _ in dict(node.inputs).items():
                if not self._is_node_exist(node_name=src_node_name):
                    print("Warning: Source node (%s) is None.", src_node_name)
                    continue

                src_node = self._get_normal_node(node_name=src_node_name)
                if src_node.type in (NodeTypeEnum.LOAD.value,
                                     NodeTypeEnum.TUPLE_GET_ITEM.value,
                                     NodeTypeEnum.MAKETUPLE.value,
                                     NodeTypeEnum.UPDATE_STATE.value):
                    node.delete_inputs(src_node_name)
                    for source_node_name, source_attr in dict(src_node.inputs).items():
                        source_node = self._get_normal_node(node_name=source_node_name)
                        if source_node is None:
                            print("Warning: Source node (%s) is None.", source_node_name)
                            continue
                        source_attr['shape'] = source_node.output_shape
                        source_attr['data_type'] = source_node.output_data_type
                        node.add_inputs(src_name=source_node.name, input_attr=source_attr)

    def _update_output_after_create_node(self):
        """Update the output of node after create node."""
        super()._update_output_after_create_node()

        nodes = self._list_nodes_without_parameter_const()
        for node in nodes:
            for src_node_name, _ in dict(node.outputs).items():
                if not self._is_node_exist(node_name=src_node_name):
                    print("Warning: Source node (%s}) is None.", src_node_name)
                    continue

                src_node = self._get_normal_node(node_name=src_node_name)
                if src_node.type in (NodeTypeEnum.LOAD.value,
                                     NodeTypeEnum.TUPLE_GET_ITEM.value,
                                     NodeTypeEnum.MAKETUPLE.value,
                                     NodeTypeEnum.UPDATE_STATE.value):
                    node.delete_outputs(src_node_name)
                    for source_node_name, source_attr in dict(src_node.outputs).items():
                        source_node = self._get_normal_node(node_name=source_node_name)
                        if source_node is None:
                            print("Warning: Source node (%s) is None.", source_node_name)
                            continue
                        source_attr['shape'] = source_node.output_shape
                        source_attr['data_type'] = source_node.output_data_type
                        node.add_outputs(src_name=source_node.name, output_attr=source_attr)

    def _delete_non_computational_ops(self):
        """Deleted non-computational operators."""
        delete_names = []
        for node in self._normal_node_map.values():
            if node.type in (NodeTypeEnum.LOAD.value,
                             NodeTypeEnum.TUPLE_GET_ITEM.value,
                             NodeTypeEnum.MAKETUPLE.value,
                             NodeTypeEnum.UPDATE_STATE.value):
                delete_names.append(node.name)

        self._delete_nodes_of_cache(delete_names)

    def _list_nodes_without_parameter_const(self):
        """List nodes without parameter and const node."""
        nodes = self._normal_node_map.values()
        not_expect_type = (NodeTypeEnum.CONST.value, NodeTypeEnum.PARAMETER.value)
        nodes = filter(lambda node: node.type not in not_expect_type, nodes)
        nodes = sorted(nodes, key=lambda node: node.topological_index)
        return nodes

    def _extract_node_by_single_node_in_scope(self):
        """Extract node from the scope which has only one node."""
        nodes = self._list_nodes_without_parameter_const()
        scope_map_types = defaultdict(set)
        scope_map_node_cnt = defaultdict(int)
        for node in nodes:
            if not node.scope or '/' not in node.scope:
                continue
            scope_map_types[node.scope].add(node.type)
            scope_map_node_cnt[node.scope] += 1

        filter_scopes = set()
        for scope, types in scope_map_types.items():
            if len(types) == 1 and scope_map_node_cnt[scope] > 1 and types.pop() in scope:
                filter_scopes.add(scope)

        for filter_scope in list(filter_scopes):
            for scope in scope_map_types:
                if scope.startswith(f'{filter_scope}/'):
                    filter_scopes.remove(filter_scope)
                    break

        if not filter_scopes:
            return

        for node in nodes:
            if node.scope in filter_scopes and '/' in node.scope:
                name = node.name.rsplit('/', 1)[1]
                new_scope = node.scope.rsplit('/', 1)[0]
                new_name = f'{new_scope}/{name}'
                self._update_node_name_of_cache(node, new_name)

        return

    def _clean_no_input_output_node(self):
        """Clean nodes which has no input and output."""
        nodes = self._list_nodes_without_parameter_const()
        deleted_names = []
        for node in nodes:
            if not node.inputs and not node.outputs:
                deleted_names.append(node.name)
        self._delete_nodes_of_cache(deleted_names)
