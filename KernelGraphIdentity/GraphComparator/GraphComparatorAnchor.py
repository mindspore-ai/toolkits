import logging
import networkx as nx
from GraphComparator.GraphComparator import GraphComparator
from ExecutingOrder.ExecutingOrder import Node


def _get_node_key(node: Node, graph: nx.DiGraph) -> str:
    """
    生成节点key值，基于当前节点特征和前置节点特征。
    """
    cur_node_feature = node.get_node_feature()
    pre_nodes = sorted(graph.predecessors(node), key=lambda x: x.get_node_feature())
    pre_nodes_features = [x.get_node_feature() for x in pre_nodes]
    return cur_node_feature + "@" + "@".join(pre_nodes_features)


def _record_key_nodes(record: dict[str, list[Node]], key: str, node: Node):
    """
    记录key值对应的节点。
    """
    record.setdefault(key, [])
    record[key].append(node)


def _record_key_times(record: dict[str, int], key: str, increment: int):
    """
    记录key值出现次数。
    """
    record.setdefault(key, 0)
    record[key] += increment


def _match_layer_nodes(
    src_graph: nx.DiGraph,
    dst_graph: nx.DiGraph,
    src_layer_nodes: list[Node],
    dst_layer_nodes: list[Node]
) -> tuple[
    dict[str, int],
    dict[str, list[Node]],
    dict[str, list[Node]]
]:
    """
    匹配层中节点，返回根据节点生成的key值出现的次数以及key值对应的节点。
    """
    key_times = {}  # 记录key值出现的次数
    src_key_nodes = {}  # 记录key值对应第一个有向图中的节点
    for node in src_layer_nodes:
        key = _get_node_key(node, src_graph)
        _record_key_times(key_times, key, 1)
        _record_key_nodes(src_key_nodes, key, node)
    
    dst_key_nodes = {}  # 记录key值对应第二个有向图中的节点
    for node in dst_layer_nodes:
        key = _get_node_key(node, dst_graph)
        _record_key_times(key_times, key, -1)
        _record_key_nodes(dst_key_nodes, key, node)

    return key_times, src_key_nodes, dst_key_nodes


def _get_layer_diff_nodes(
    src_graph: nx.DiGraph,
    dst_graph: nx.DiGraph,
    src_layer_nodes: list[Node],
    dst_layer_nodes: list[Node],
    src_mismatch_nodes: set[Node],
    dst_mismatch_nodes: set[Node]
) -> bool:
    """
    获取当前层中的差异节点。
    """
    key_times, src_key_nodes, dst_key_nodes = _match_layer_nodes(src_graph, dst_graph, src_layer_nodes, dst_layer_nodes)
    found = False
    for key, times in key_times.items():
        if times == 0:
            continue
        logging.info(f"Diff node key {key}.")
        found = True
        if src_key_nodes.get(key):
            src_mismatch_nodes.update(src_key_nodes[key])
        if dst_key_nodes.get(key):
            dst_mismatch_nodes.update(dst_key_nodes[key])
    return found


class GraphComparatorAnchor(GraphComparator):
    """
    基于锚点比较两有向图的差异。
    """
    def __init__(
        self,
        src_graph: nx.DiGraph,
        dst_graph: nx.DiGraph,
        src_topo_layers: list[list[Node]],
        dst_topo_layers: list[list[Node]]
    ):
        super().__init__(src_graph, dst_graph)
        self._src_topo_layers = src_topo_layers
        self._dst_topo_layers = dst_topo_layers

    def compare_graphs(self) -> tuple[set[Node], set[Node], set[Node], set[Node]]:
        src_topo_layers = self._src_topo_layers
        dst_topo_layers = self._dst_topo_layers
        src_graph = self._src_graph
        dst_graph = self._dst_graph
        src_match_nodes = set()  # 保存第一个有向图已匹配的节点
        dst_match_nodes = set()  # 保存第二个有向图已匹配的节点
        layer_id = 0
        while True:
            if layer_id == len(src_topo_layers) and layer_id == len(dst_topo_layers):
                # 所有节点都匹配
                return src_match_nodes, dst_match_nodes, set(), set()
            if layer_id == len(src_topo_layers):
                # 第一个有向图所有节点都匹配，第二个有向图剩余节点都不匹配，只输出一层不匹配的节点
                return src_match_nodes, dst_match_nodes, set(), set(dst_topo_layers[layer_id])
            if layer_id == len(dst_topo_layers):
                # 第二个有向图所有节点都匹配，第一个有向图剩余节点都不匹配，只输出一层不匹配的节点
                return src_match_nodes, dst_match_nodes, set(src_topo_layers[layer_id]), set()
            src_layer_nodes = src_topo_layers[layer_id]
            dst_layer_nodes = dst_topo_layers[layer_id]
            src_mismatch_nodes = set()
            dst_mismatch_nodes = set()
            if _get_layer_diff_nodes(src_graph, dst_graph,
                                     src_layer_nodes, dst_layer_nodes,
                                     src_mismatch_nodes, dst_mismatch_nodes):
                # 找到了存在差异的层，输出该层中不匹配的节点
                src_match_nodes.update(set(src_layer_nodes) - src_mismatch_nodes)
                dst_match_nodes.update(set(dst_layer_nodes) - dst_mismatch_nodes)
                return src_match_nodes, dst_match_nodes, src_mismatch_nodes, dst_mismatch_nodes
            src_match_nodes.update(src_layer_nodes)
            dst_match_nodes.update(dst_layer_nodes)
            layer_id += 1
