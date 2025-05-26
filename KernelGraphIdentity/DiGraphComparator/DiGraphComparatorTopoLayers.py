import networkx as nx
from DiGraphComparator.DiGraphComparator import DiGraphComparator
from ExecutingOrder.ExecutingOrder import Node


class DiGraphComparatorTopoLayers(DiGraphComparator):
    """
    基于拓扑分层的有向图差异比较器。
    """
    def __init__(self, src_graph: nx.DiGraph, dst_graph: nx.DiGraph, src_anchor: Node, dst_anchor: Node):
        super().__init__(src_graph, dst_graph)
        self._src_anchor = src_anchor
        self._dst_anchor = dst_anchor
        self._is_compared = False
        self._up_direction = True  # 为True时，从底层向顶层比较，为False时，从顶层向底层比较，默认从底层向顶层比较
        self._src_match_nodes: set[Node] = set()
        self._dst_match_nodes: set[Node] = set()
        self._src_mismatch_nodes: set[Node] = set()
        self._dst_mismatch_nodes: set[Node] = set()
        self._src_mismatch_nodes_k: dict[Node, str] = {}
        self._dst_mismatch_nodes_k: dict[Node, str] = {}
        self._src_mismatch_k_nodes: dict[str, set[Node]] = {}
        self._dst_mismatch_k_nodes: dict[str, set[Node]] = {}
        self._has_diff = True  # 是否存在差异

    def set_src_anchor(self, src_anchor: Node):
        self._src_anchor = src_anchor

    def set_dst_anchor(self, dst_anchor: Node):
        self._dst_anchor = dst_anchor

    def get_src_anchor(self) -> Node:
        return self._src_anchor

    def get_dst_anchor(self) -> Node:
        return self._dst_anchor

    def is_compared(self) -> bool:
        return self._is_compared

    def get_up_direction(self) -> bool:
        return self._up_direction

    def src_node_is_matched(self, node: Node) -> bool:
        return node in self._src_match_nodes

    def dst_node_is_matched(self, node: Node) -> bool:
        return node in self._dst_match_nodes

    def src_node_is_mismatched(self, node: Node) -> bool:
        return node in self._src_mismatch_nodes

    def dst_node_is_mismatched(self, node: Node) -> bool:
        return node in self._dst_mismatch_nodes

    @staticmethod
    def _get_node_key(node: Node, graph: nx.DiGraph, up_direction: bool) -> str:
        """
        生成节点key值，基于当前节点特征和前置节点特征。
        """
        get_around = graph.successors
        if up_direction:
            get_around = graph.predecessors

        cur_node_feature = node.get_node_feature()
        around_nodes = sorted(get_around(node), key=lambda around_node: around_node.get_node_feature())
        around_nodes_features = [around_node.get_node_feature() for around_node in around_nodes]
        return cur_node_feature + "@" + "@".join(around_nodes_features)

    @staticmethod
    def _record_key_nodes(record: dict[str, set[Node]], key: str, node: Node):
        """
        记录key值对应的节点。
        """
        record.setdefault(key, set())
        record[key].add(node)

    @staticmethod
    def _record_key_times(record: dict[str, int], key: str):
        """
        记录key值出现次数。
        """
        record.setdefault(key, 0)
        record[key] += 1

    def _match_layer_nodes(
            self,
            src_layer_nodes: list[Node],
            dst_layer_nodes: list[Node]
    ) -> tuple[
        dict[str, int],
        dict[str, int],
        dict[str, set[Node]],
        dict[str, set[Node]]
    ]:
        """
        匹配层中节点，返回根据节点生成的key值出现的次数以及key值对应的节点。
        """
        up_direction = self._up_direction
        src_graph = self._src_graph
        dst_graph = self._dst_graph

        src_key_times: dict[str, int] = {}  # 记录key值出现的次数
        src_key_nodes: dict[str, set[Node]] = {}  # 记录key值对应第一个有向图中的节点
        for node in src_layer_nodes:
            key = DiGraphComparatorTopoLayers._get_node_key(node, src_graph, up_direction)
            DiGraphComparatorTopoLayers._record_key_times(src_key_times, key)
            DiGraphComparatorTopoLayers._record_key_nodes(src_key_nodes, key, node)

        dst_key_times = {}
        dst_key_nodes = {}
        for node in dst_layer_nodes:
            key = DiGraphComparatorTopoLayers._get_node_key(node, dst_graph, up_direction)
            DiGraphComparatorTopoLayers._record_key_times(dst_key_times, key)
            DiGraphComparatorTopoLayers._record_key_nodes(dst_key_nodes, key, node)

        return src_key_times, dst_key_times, src_key_nodes, dst_key_nodes

    def _get_layer_diff_nodes(self, src_layer_nodes: list[Node], dst_layer_nodes: list[Node]) -> bool:
        """
        获取当前层中的差异节点。
        """
        src_key_times, dst_key_times, src_key_nodes, dst_key_nodes = self._match_layer_nodes(
            src_layer_nodes,
            dst_layer_nodes
        )
        found = False
        for key in set(src_key_times.keys()) | set(dst_key_times.keys()):
            if dst_key_times.get(key) is None:  # key值只在src图中存在
                self._src_mismatch_nodes.update(src_key_nodes[key])
                for node in src_key_nodes[key]:
                    self._src_mismatch_nodes_k[node] = key
                self._src_mismatch_k_nodes[key] = src_key_nodes[key]
                found = True
                continue
            if src_key_times.get(key) is None:  # key值只在dst图中存在
                self._dst_mismatch_nodes.update(dst_key_nodes[key])
                for node in dst_key_nodes[key]:
                    self._dst_mismatch_nodes_k[node] = key
                self._dst_mismatch_k_nodes[key] = dst_key_nodes[key]
                found = True
                continue
            if src_key_times[key] != dst_key_times[key]:  # key值在两图中出现次数不一致
                self._src_mismatch_nodes.update(src_key_nodes[key])
                for node in src_key_nodes[key]:
                    self._src_mismatch_nodes_k[node] = key
                self._src_mismatch_k_nodes[key] = src_key_nodes[key]
                self._dst_mismatch_nodes.update(dst_key_nodes[key])
                for node in dst_key_nodes[key]:
                    self._dst_mismatch_nodes_k[node] = key
                self._dst_mismatch_k_nodes[key] = dst_key_nodes[key]
                found = True
        return found

    def compare_graphs(self, up_direction: bool, src_topo_layers: list[list[Node]], dst_topo_layers: list[list[Node]]):
        """
        要求流向为向下进行拓扑分层，索引值小的为底层。
        """
        if self._is_compared:
            return
        self._up_direction = up_direction
        if not self._up_direction:
            src_topo_layers = src_topo_layers[::-1]
            dst_topo_layers = dst_topo_layers[::-1]
        src_topo_layer_num = len(src_topo_layers)
        dst_topo_layer_num = len(dst_topo_layers)
        layer_id = 0
        while True:
            if layer_id == src_topo_layer_num and layer_id == dst_topo_layer_num:
                """
                所有节点都匹配
                """
                self._has_diff = False
                break
            src_layer_nodes = src_topo_layers[layer_id]
            dst_layer_nodes = dst_topo_layers[layer_id]
            if self._get_layer_diff_nodes(src_layer_nodes, dst_layer_nodes):
                """
                找到差异层
                """
                self._src_match_nodes.update(set(src_layer_nodes) - self._src_mismatch_nodes)
                self._dst_match_nodes.update(set(dst_layer_nodes) - self._dst_mismatch_nodes)
                break
            self._src_match_nodes.update(src_layer_nodes)
            self._dst_match_nodes.update(dst_layer_nodes)
            layer_id += 1
        self._is_compared = True

    def clear_compared_result(self):
        self._is_compared = False
        self._up_direction = True
        self._src_match_nodes.clear()
        self._dst_match_nodes.clear()
        self._src_mismatch_nodes.clear()
        self._dst_mismatch_nodes.clear()
        self._src_mismatch_nodes_k.clear()
        self._dst_mismatch_nodes_k.clear()
        self._src_mismatch_k_nodes.clear()
        self._dst_mismatch_k_nodes.clear()
        self._has_diff = True

    @staticmethod
    def _get_node_operator_type_and_in_out_memory_attibutes_info(node: Node, padding: str) -> list[str]:
        info_lines: list[str] = []

        info_lines.append(padding + f"算子类型：{node.get_operator_type()}")

        input_memory_attributes = list(str(memory_attribute)
                                       for memory_attribute in node.get_input_memory_attributes())
        input_memory_attributes_str = ", ".join(input_memory_attributes)
        info_lines.append(padding + f"输入数量：{len(input_memory_attributes)}")
        info_lines.append(padding + f"输入属性：{input_memory_attributes_str}")

        output_memory_attributes = list(str(memory_attribute)
                                        for memory_attribute in node.get_output_memory_attributes())
        output_memory_attributes_str = ", ".join(output_memory_attributes)
        info_lines.append(padding + f"输出数量：{len(output_memory_attributes)}")
        info_lines.append(padding + f"输出属性：{output_memory_attributes_str}")

        return info_lines

    def _get_node_feature_info(self, node: Node, padding: str, graph: nx.DiGraph) -> list[str]:
        info_lines: list[str] = []

        info_lines.append(padding + "本节点特征：")

        info_lines.extend(
            DiGraphComparatorTopoLayers._get_node_operator_type_and_in_out_memory_attibutes_info(
                node,
                padding + " " * 2
            )
        )

        if self._up_direction:
            info_lines.append(padding + " " * 2 + "前驱算子：")
            get_around = graph.predecessors
        else:
            info_lines.append(padding + " " * 2 + "后继算子：")
            get_around = graph.successors

        around_nodes = sorted(get_around(node), key=lambda around_node: around_node.get_node_feature())
        for around_node in around_nodes:
            info_lines.extend(
                DiGraphComparatorTopoLayers._get_node_operator_type_and_in_out_memory_attibutes_info(
                    around_node,
                    padding + " " * 4
                )
            )
            info_lines.append("")

        return info_lines

    def get_src_mismatch_node_info(self, node: Node, padding: str) -> list[str]:
        info_lines: list[str] = []

        info_lines.extend(self._get_node_feature_info(node, padding, self._src_graph))

        node_k = self._src_mismatch_nodes_k[node]
        src_k_nodes = self._src_mismatch_k_nodes[node_k]
        info_lines.append(padding + "差异原因为本图与对端图本层该特征算子数量不一样")
        info_lines.append(padding + f"本图本层该特征算子数量为：{len(src_k_nodes)}")
        if self._dst_mismatch_k_nodes.get(node_k) is None:
            info_lines.append(padding + f"对端图本层该特征算子数量为：0")
        else:
            dst_k_nodes = self._dst_mismatch_k_nodes[node_k]
            info_lines.append(padding + f"对端图本层该特征算子数量为：{len(dst_k_nodes)}")

        info_lines.append(padding + "本图本层该特征的算子有：")
        for node in src_k_nodes:
            info_lines.append(padding + " " * 2 + f"算子类型 {node.get_operator_type()} 行号 {node.get_line_num()}")

        if self._dst_mismatch_k_nodes.get(node_k) is not None:
            dst_k_nodes = self._dst_mismatch_k_nodes[node_k]
            info_lines.append(padding + "对端图本层该特征的算子有：")
            for node in dst_k_nodes:
                info_lines.append(padding + " " * 2 + f"算子类型 {node.get_operator_type()} 行号 {node.get_line_num()}")

        return info_lines

    def get_dst_mismatch_node_info(self, node: Node, padding: str) -> list[str]:
        info_lines: list[str] = []

        info_lines.extend(self._get_node_feature_info(node, padding, self._dst_graph))

        node_k = self._dst_mismatch_nodes_k[node]
        dst_k_nodes = self._dst_mismatch_k_nodes[node_k]
        info_lines.append(padding + "差异原因为本图与对端图本层该特征算子数量不一样")
        info_lines.append(padding + f"本图本层该特征算子数量为：{len(dst_k_nodes)}")
        if self._src_mismatch_k_nodes.get(node_k) is None:
            info_lines.append(padding + "对端图本层该特征算子数量为：0")
        else:
            src_k_nodes = self._src_mismatch_k_nodes.get(node_k)
            info_lines.append(padding + f"对端图本层该特征算子数量为：{len(src_k_nodes)}")

        info_lines.append(padding + "本图本层该特征的算子有：")
        for node in dst_k_nodes:
            info_lines.append(padding + " " * 2 + f"算子类型: {node.get_operator_type()} 行号 {node.get_line_num()}")

        if self._src_mismatch_k_nodes.get(node_k) is not None:
            src_k_nodes = self._src_mismatch_k_nodes.get(node_k)
            info_lines.append(padding + "对端图本层该特征的算子有：")
            for node in src_k_nodes:
                info_lines.append(padding + " " * 2 + f"算子类型: {node.get_operator_type()} 行号 {node.get_line_num()}")

        return info_lines
