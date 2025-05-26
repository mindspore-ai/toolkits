import copy
import re
import networkx as nx
from ExecutingOrder.MemoryOperatedRecorder import MemoryOperatedRecorder
from ExecutingOrder.EdgeManager import EdgeManager
from ExecutingOrder.MemoryAttribute import MemoryAttribute
from ExecutingOrder.NodeManager import SINK_VIRTUAL_NODE_OPERATOR_TYPE, SOURCE_VIRTUAL_NODE_OPERATOR_TYPE
from ExecutingOrder.NodeManager import Node, NodeManager


class ExecutingOrder:
    def __init__(self, file_path: str):
        self._graph = nx.DiGraph()
        self._node_manager: NodeManager = NodeManager()
        self._edge_manager: EdgeManager = EdgeManager()
        self._memory_operated_recorder: MemoryOperatedRecorder = MemoryOperatedRecorder()

        self._get_graph_by_file(file_path)

    def get_graph(self):
        return self._graph

    def get_node_manager(self) -> NodeManager:
        return self._node_manager

    def get_edge_manager(self) -> EdgeManager:
        return self._edge_manager

    @staticmethod
    def _get_in_out_memory_ids_and_operator(file_data: dict, in_out_memory_ids_and_operator_str: str, line_id: int):
        # (%1445) = Concat(%1441, %1384), task_info: 1444, attrs {stream_id=0}
        in_out_memory_ids_and_operator_pattern = r"\((.*)\) = (.*)\((.*)\), task_info: (\d+), attrs \{(.*)\}"
        in_out_memory_ids_and_operator_match = re.findall(in_out_memory_ids_and_operator_pattern,
                                                          in_out_memory_ids_and_operator_str)
        in_out_memory_ids_and_operator = in_out_memory_ids_and_operator_match[0]

        output_memory_ids = in_out_memory_ids_and_operator[0]
        output_memory_ids = output_memory_ids.replace(" ", "").split(",") if output_memory_ids != "" else []

        operator_type = in_out_memory_ids_and_operator[1]
        # 过滤cell信息，获取算子类型
        operator_type_pattern = r"(.*)/([^/-]+)-"
        operator_type_match = re.findall(operator_type_pattern, operator_type)
        full_scope = []
        if len(operator_type_match) != 0:
            full_scope = operator_type_match[0][0].split("/")
            operator_type = operator_type_match[0][1]

        input_memory_ids = in_out_memory_ids_and_operator[2]
        input_memory_ids = input_memory_ids.replace(" ", "").split(",") if input_memory_ids != "" else []

        task_info = in_out_memory_ids_and_operator[3]

        attrs = in_out_memory_ids_and_operator[4]

        file_data["output_memory_ids"] = output_memory_ids
        file_data["operator_type"] = operator_type
        file_data["full_scope"] = full_scope
        file_data["input_memory_ids"] = input_memory_ids
        file_data["task_info"] = task_info
        file_data["attrs"] = attrs
        file_data["line_num"] = line_id + 1

    @staticmethod
    def _get_stride_offset(stride_offset_str: str, memory_attribute: MemoryAttribute):
        if stride_offset_str == "":
            return

        memory_attribute.set_exist_stride_offset()

        # strdes=[12288, 12288, 384, 1],offset=0
        stride_offset_pattern = r"strdes=\[([\d, ]*)\],offset=(\d*)"
        stride_offset_match = re.findall(stride_offset_pattern, stride_offset_str)
        stride_offset = stride_offset_match[0]
        if stride_offset[0] != "":
            stride = stride_offset[0].split(",")
            memory_attribute.set_stride([int(x.strip()) for x in stride])
        memory_attribute.set_offset(int(stride_offset[1]))

    @staticmethod
    def _get_memory_attributes(memory_attributes_str: str) -> list[MemoryAttribute]:
        if memory_attributes_str == "":
            return []

        # BFloat16:[4096, 1, 32, 128]{strdes=[12288, 12288, 384, 1],offset=0}
        memory_attribute_pattern = r"(\w*):\[([\d, ]*)\](?:\{(.*?)\})?"
        memory_attribute_matchs = re.findall(memory_attribute_pattern, memory_attributes_str)
        memory_attributes = []
        for memory_attribute_match in memory_attribute_matchs:
            memory_attribute = MemoryAttribute()
            memory_attribute.set_dtype(memory_attribute_match[0])
            if memory_attribute_match[1] != "":
                shape = memory_attribute_match[1].split(",")
                memory_attribute.set_shape([int(x.strip()) for x in shape])
            ExecutingOrder._get_stride_offset(memory_attribute_match[2], memory_attribute)
            memory_attributes.append(memory_attribute)

        return memory_attributes

    @staticmethod
    def _get_in_out_memory_attributes(file_data: dict, in_out_memory_attributes_str: str):
        # (BFloat16:[4096, 1, 4096]{strdes=[4096, 4096, 1],offset=0}) <- (BFloat16:[4096, 1, 32, 128])
        in_out_memory_attributes_pattern = r"\((.*)\) <- \((.*)\)"
        in_out_memory_attributes_match = re.findall(in_out_memory_attributes_pattern, in_out_memory_attributes_str)
        in_out_memory_attributes = in_out_memory_attributes_match[0]
        out_memory_attributes_str, in_memory_attributes_str = in_out_memory_attributes[0], in_out_memory_attributes[1]

        file_data["output_memory_attributes"] = ExecutingOrder._get_memory_attributes(out_memory_attributes_str)
        file_data["input_memory_attributes"] = ExecutingOrder._get_memory_attributes(in_memory_attributes_str)

    @staticmethod
    def _get_file_datas(file_path: str) -> list[dict]:
        with open(file_path, "r") as f:
            lines = f.readlines()

        file_datas = []
        for line_id in range(0, len(lines) - 1, 3):
            file_data = {}

            in_out_memory_ids_and_operator_str = lines[line_id]
            ExecutingOrder._get_in_out_memory_ids_and_operator(file_data, in_out_memory_ids_and_operator_str, line_id)

            in_out_memory_attributes_str = lines[line_id + 1]
            ExecutingOrder._get_in_out_memory_attributes(file_data, in_out_memory_attributes_str)

            stack_str = lines[line_id + 2]
            file_data["stack"] = stack_str.split("|")

            file_datas.append(file_data)

        return file_datas

    def _add_nodes(self, file_datas: list[dict]):
        for file_data in file_datas:
            node = Node()
            node.set_operator_type(file_data["operator_type"])
            node.set_full_scope(file_data["full_scope"])
            node.set_stack(file_data["stack"])
            node.set_input_memory_ids(file_data["input_memory_ids"])
            node.set_output_memory_ids(file_data["output_memory_ids"])
            node.set_input_memory_attributes(file_data["input_memory_attributes"])
            node.set_output_memory_attributes(file_data["output_memory_attributes"])
            node.set_task_info(file_data["task_info"])
            node.set_attrs(file_data["attrs"])
            node.set_line_num(file_data["line_num"])

            self._node_manager.add_node(node)
            self._graph.add_node(node)

    @staticmethod
    def _get_build_edge_key(memory_id: str, memory_attribute: MemoryAttribute):
        return f"{memory_id}+{memory_attribute}"

    @staticmethod
    def _can_build_edge(
            memory_id: str,
            memory_attribute: MemoryAttribute,
            predecessors: dict[str, Node]
    ) -> bool:
        build_edge_key = ExecutingOrder._get_build_edge_key(memory_id, memory_attribute)
        if predecessors.get(build_edge_key) is None:
            return False
        return True

    @staticmethod
    def _get_build_edge_predecessor(
            memory_id: str,
            memory_attribute: MemoryAttribute,
            predecessors: dict[str, Node]
    ) -> Node:
        build_edge_key = ExecutingOrder._get_build_edge_key(memory_id, memory_attribute)
        return predecessors[build_edge_key]

    @staticmethod
    def _update_build_edge_predecessors(
            memory_id: str,
            memory_attribute: MemoryAttribute,
            predecessors: dict[str, Node],
            predecessor: Node
    ):
        build_edge_key = ExecutingOrder._get_build_edge_key(memory_id, memory_attribute)
        predecessors[build_edge_key] = predecessor

    def _add_edges(self):
        build_edge_predecessors: dict[str, Node] = {}  # 记录前驱
        for node in self._node_manager.get_all_nodes():
            input_memory_ids = node.get_input_memory_ids()
            if input_memory_ids:
                for input_memory_id, input_memory_attribute in zip(input_memory_ids,
                                                                   node.get_input_memory_attributes()):
                    # self._memory_operated_recorder.record_operation(
                    #     input_memory_id,
                    #     node,
                    #     input_memory_attribute,
                    #     "input"
                    # )
                    can_build_edge = ExecutingOrder._can_build_edge(
                        input_memory_id,
                        input_memory_attribute,
                        build_edge_predecessors
                    )
                    if not can_build_edge:
                        continue
                    predecessor = ExecutingOrder._get_build_edge_predecessor(
                        input_memory_id,
                        input_memory_attribute,
                        build_edge_predecessors
                    )
                    self._edge_manager.update_edge(predecessor, node, input_memory_id, [input_memory_attribute])
                    self._graph.add_edge(predecessor, node)  # DiGraph有重复边添加时会自动忽略，无需额外处理重复边

            output_memory_ids = node.get_output_memory_ids()
            if output_memory_ids:
                for output_memory_id, output_memory_attribute in zip(output_memory_ids,
                                                                     node.get_output_memory_attributes()):
                    # self._memory_operated_recorder.record_operation(
                    #     output_memory_id,
                    #     node,
                    #     output_memory_attribute,
                    #     "output"
                    # )
                    ExecutingOrder._update_build_edge_predecessors(
                        output_memory_id,
                        output_memory_attribute,
                        build_edge_predecessors,
                        node
                    )

    @staticmethod
    def _need_ignore(node: Node) -> bool:
        ignore_operator_types = [
            "data"
        ]
        if node.get_operator_type() in ignore_operator_types:
            return True

        return False

    def _remove_memory_ids_for_ignore_nodes(self, node: Node):
        """
        data节点移除前，先修改data节点后继节点的输入
        """
        graph = self._graph
        output_memory_id = node.get_output_memory_ids()[0]
        output_memory_attribute = node.get_output_memory_attributes()[0]
        for successor in graph.successors(node):
            need_remove_ids = []
            input_memory_ids = successor.get_input_memory_ids()
            input_memory_attributes = successor.get_input_memory_attributes()
            for i in range(len(input_memory_ids)):
                if output_memory_id == input_memory_ids[i] and output_memory_attribute == input_memory_attributes[i]:
                    need_remove_ids.append(i)
            for need_remove_id in need_remove_ids[::-1]:
                del input_memory_ids[need_remove_id]
                del input_memory_attributes[need_remove_id]

    def _ignore_nodes(self):
        """
        只支持data节点
        """
        graph = self._graph
        for node in self._node_manager.get_all_nodes():
            if not ExecutingOrder._need_ignore(node):
                continue
            if len(node.get_output_memory_ids()) == 0:
                graph.remove_node(node)
                continue
            self._remove_memory_ids_for_ignore_nodes(node)
            graph.remove_node(node)

    def _record_isolated_nodes(self):
        nodes_degree = {node: 0 for node in self._graph.nodes()}
        for src, dst in self._graph.edges():
            nodes_degree[src] += 1
            nodes_degree[dst] += 1
        nodes_to_remove = [node for node in self._graph.nodes() if nodes_degree[node] == 0]
        for node in nodes_to_remove:
            self._graph.remove_node(node)
            self._node_manager.record_isolated_node(node)

    @staticmethod
    def _set_node_feature(node: Node):
        input_memory_attributes = list(str(memory_attribute)
                                       for memory_attribute in node.get_input_memory_attributes())
        input_memory_attributes_str = ", ".join(input_memory_attributes)

        output_memory_attributes = list(str(memory_attribute)
                                        for memory_attribute in node.get_output_memory_attributes())
        output_memory_attributes_str = ", ".join(output_memory_attributes)

        node_feature = f"({output_memory_attributes_str})<{node.get_operator_type()}-({input_memory_attributes_str})"
        node.set_node_feature(node_feature)

        # node.set_node_feature(node.get_operator_type())

    def _set_nodes_feature(self):
        for node in self._node_manager.get_all_nodes():
            ExecutingOrder._set_node_feature(node)

    def _set_first_node(self):
        nodes_in_degree = {node: 0 for node in self._graph.nodes()}
        for _, dst in self._graph.edges():
            nodes_in_degree[dst] += 1
        for node in self._graph.nodes():
            if nodes_in_degree[node] == 0:
                self._node_manager.set_first_node(node)
                break

    def _add_source_virtual_node(self):
        virtual_node = self._node_manager.get_source_virtual_node()
        # 将虚拟节点加入图中
        self._graph.add_node(virtual_node)
        # 计算图中所有节点的入度
        nodes_in_degree = {node: 0 for node in self._graph.nodes()}
        for _, dst in self._graph.edges():
            nodes_in_degree[dst] += 1
        # 创建虚拟节点到入度为0节点的边
        for node in self._graph.nodes():
            if nodes_in_degree[node] != 0:
                continue
            if node.get_operator_type() == SOURCE_VIRTUAL_NODE_OPERATOR_TYPE:  # 不需要和自己创建边
                continue
            self._graph.add_edge(virtual_node, node)
            self._edge_manager.add_edge(virtual_node, node)

    def _add_sink_virtual_node(self):
        virtual_node = self._node_manager.get_sink_virtual_node()
        # 将虚拟节点加入图中
        self._graph.add_node(virtual_node)
        # 计算图中所有节点的出度
        nodes_out_degree = {node: 0 for node in self._graph.nodes()}
        for src, _ in self._graph.edges():
            nodes_out_degree[src] += 1
        # 创建出度为0节点到虚拟节点的边
        for node in self._graph.nodes():
            if nodes_out_degree[node] != 0:
                continue
            if node.get_operator_type() == SINK_VIRTUAL_NODE_OPERATOR_TYPE:  # 不需要和自己创建边
                continue
            self._graph.add_edge(node, virtual_node)
            self._edge_manager.add_edge(node, virtual_node)

    def _add_virtual_nodes(self):
        # self._add_source_virtual_node()
        self._add_sink_virtual_node()

    def _remove_virtual_nodes(self):
        # virtual_node = self._node_manager.get_source_virtual_node()
        # self._graph.remove_node(virtual_node)

        virtual_node = self._node_manager.get_sink_virtual_node()
        self._graph.remove_node(virtual_node)

    def _get_graph_by_file_datas(self, file_datas: list[dict]):
        self._add_nodes(file_datas)
        self._add_edges()
        self._ignore_nodes()
        self._record_isolated_nodes()
        self._set_nodes_feature()
        self._set_first_node()
        self._add_virtual_nodes()

    def _get_graph_by_file(self, file_path: str):
        file_datas = ExecutingOrder._get_file_datas(file_path)
        self._get_graph_by_file_datas(file_datas)

    @staticmethod
    def _has_source_to_source_cycle(graph: nx.DiGraph, source: Node) -> bool:
        """
        判断从 source 节点出发是否能回到 source 节点（存在环）
        """
        visited = set()
        stack = [(source, iter(graph.successors(source)))]
        while stack:
            _, successors = stack[-1]
            try:
                successor = next(successors)
            except StopIteration:
                stack.pop()
                continue

            if successor == source:
                return True

            if successor in visited:
                continue

            visited.add(successor)
            stack.append((successor, iter(graph.successors(successor))))

        return False

    def fuse_nodes_cycle_check(self, fuse_nodes_id: list[int]) -> bool:
        fuse_nodes = {self._node_manager.get_node_by_node_id(node_id) for node_id in fuse_nodes_id}

        # 复制原图，用新图来检查融合后的图是否会形成环
        new_graph = copy.deepcopy(self._graph)

        new_fuse_nodes = set()
        for fuse_node in fuse_nodes:
            for node in new_graph.nodes():
                if node == fuse_node:
                    new_fuse_nodes.add(node)
                    break

        successors: set[Node] = set()
        predecessors: set[Node] = set()
        for node in new_fuse_nodes:
            for successor in new_graph.successors(node):
                if successor not in new_fuse_nodes:
                    successors.add(successor)
            for predecessor in new_graph.predecessors(node):
                if predecessor not in new_fuse_nodes:
                    predecessors.add(predecessor)

        new_graph.remove_nodes_from(new_fuse_nodes)

        fused_node = Node()
        new_graph.add_node(fused_node)
        for successor in successors:
            new_graph.add_edge(fused_node, successor)
        for predecessor in predecessors:
            new_graph.add_edge(predecessor, fused_node)

        return ExecutingOrder._has_source_to_source_cycle(new_graph, fused_node)

    @staticmethod
    def _set_fused_node_input_output_memory_ids_and_memory_attributes(fused_node: Node, peer_node: Node):
        fused_node.set_input_memory_ids(copy.deepcopy(peer_node.get_input_memory_ids()))
        fused_node.set_output_memory_ids(copy.deepcopy(peer_node.get_output_memory_ids()))
        fused_node.set_input_memory_attributes(copy.deepcopy(peer_node.get_input_memory_attributes()))
        fused_node.set_output_memory_attributes(copy.deepcopy(peer_node.get_output_memory_attributes()))

    def _create_fused_node(self, peer_node: Node, fused_node_operator_type: str) -> Node:
        fused_node = Node()
        self._node_manager.add_node(fused_node)

        fused_node.set_operator_type(fused_node_operator_type)
        if fused_node_operator_type == "":
            fused_node.set_operator_type(peer_node.get_operator_type())

        ExecutingOrder._set_fused_node_input_output_memory_ids_and_memory_attributes(fused_node, peer_node)
        ExecutingOrder._set_node_feature(fused_node)

        return fused_node

    def _get_fused_node_suc_pre_and_old_nodes(
            self,
            fuse_nodes_id: list[int]
    ) -> tuple[
        dict[Node, list[Node]],
        dict[Node, list[Node]]
    ]:
        graph = self._graph
        successors_old_nodes: dict[Node, list[Node]] = {}
        predecessors_old_nodes: dict[Node, list[Node]] = {}
        for node_id in fuse_nodes_id:
            node = self._node_manager.get_node_by_node_id(node_id)
            for successor in graph.successors(node):
                if successor.get_node_id() in fuse_nodes_id:
                    continue
                successors_old_nodes.setdefault(successor, []).append(node)
            for predecessor in graph.predecessors(node):
                if predecessor.get_node_id() in fuse_nodes_id:
                    continue
                predecessors_old_nodes.setdefault(predecessor, []).append(node)
        return successors_old_nodes, predecessors_old_nodes

    def _add_fused_node_to_graph(
            self,
            fused_node: Node,
            successors_old_nodes: dict[Node, list[Node]],
            predecessors_old_nodes: dict[Node, list[Node]]
    ):
        graph = self._graph

        graph.add_node(fused_node)

        for successor in successors_old_nodes.keys():
            graph.add_edge(fused_node, successor)
        for predecessor in predecessors_old_nodes.keys():
            graph.add_edge(predecessor, fused_node)

    def fuse_nodes(self, fuse_nodes_id: list[int], peer_node: Node, fused_node_operator_type: str) -> Node:
        graph = self._graph
        # 创建融合后新节点
        fused_node = self._create_fused_node(peer_node, fused_node_operator_type)
        # 获取融合后新节点的前驱节点和后继节点，并记录这些前驱节点和后继节点原来连接的融合前节点
        successors_old_nodes, predecessors_old_nodes = self._get_fused_node_suc_pre_and_old_nodes(fuse_nodes_id)
        # 删除融合前节点
        graph.remove_nodes_from(self._node_manager.get_node_by_node_id(node_id) for node_id in fuse_nodes_id)
        # 将融合后新节点加入图中
        self._add_fused_node_to_graph(fused_node, successors_old_nodes, predecessors_old_nodes)
        self._edge_manager.add_edge_for_fuse(fused_node, successors_old_nodes, predecessors_old_nodes)

        return fused_node

    def delete_nodes(self, nodes_id: list[int]):
        graph = self._graph
        graph.remove_nodes_from(self._node_manager.get_node_by_node_id(node_id) for node_id in nodes_id)
        self._remove_virtual_nodes()
        self._add_virtual_nodes()

    def delete_edges(self, edges: list[dict[str, int]]):
        graph = self._graph
        for edge in edges:
            src_node_id = edge["src_node_id"]
            dst_node_id = edge["dst_node_id"]
            src_node = self._node_manager.get_node_by_node_id(src_node_id)
            dst_node = self._node_manager.get_node_by_node_id(dst_node_id)
            graph.remove_edge(src_node, dst_node)
        self._remove_virtual_nodes()
        self._add_virtual_nodes()
