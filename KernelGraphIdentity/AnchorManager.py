import logging
import networkx as nx
from ExecutingOrder.ExecutingOrder import ExecutingOrder
from ExecutingOrder.NodeManager import VIRTUAL_NODE_OPERATOR_TYPE, Node


class NodePosition:
    def __init__(self, abs_x: int, abs_y: int, rel_x: int, rel_y: int):
        self.abs_x = abs_x
        self.abs_y = abs_y
        self.rel_x = rel_x
        self.rel_y = rel_y


# 根据anchor获取周边子图
def _get_anchor_around_sub_graph(graph: nx.DiGraph, anchor: Node, down_or_up: bool) -> nx.DiGraph:
    """
    down_or_up为True表示向下生长，为False表示向上生长
    """
    get_around = graph.successors
    if not down_or_up:
        get_around = graph.predecessors

    sub_graph_nodes = {anchor}
    cur_layer_nodes = {anchor}
    while True:
        around_nodes = set()
        for cur_layer_node in cur_layer_nodes:
            around_nodes.update(get_around(cur_layer_node))
        around_nodes = around_nodes - sub_graph_nodes
        if len(around_nodes) == 0:
            logging.info(f"Get anchor around sub graph, node num is {len(sub_graph_nodes)}.")
            break
        sub_graph_nodes.update(around_nodes)
        cur_layer_nodes = around_nodes

    return graph.subgraph(sub_graph_nodes)


def _get_nodes_degree(graph: nx.DiGraph, down_or_up: bool) -> dict[Node, int]:
    """
    down_or_up为True表示向下生长，为False表示向上生长
    """
    nodes_degree = {node: 0 for node in graph.nodes()}
    if down_or_up:
        for _, dst in graph.edges():
            nodes_degree[dst] += 1
    else:
        for src, _ in graph.edges():
            nodes_degree[src] += 1
    return nodes_degree


def _get_topo_layers(graph: nx.DiGraph, down_or_up: bool) -> list[list[Node]]:
    """
    down_or_up为True表示向下生长，为False表示向上生长
    """
    nodes_degree = _get_nodes_degree(graph, down_or_up)

    if down_or_up:
        get_around = graph.successors
    else:
        get_around = graph.predecessors

    cur_layer = [node for node, degree in nodes_degree.items() if degree == 0]
    cur_layer_index = 0
    topo_layers = [cur_layer]
    while True:
        if len(cur_layer) == 0:
            logging.info(f"Max layer num is {cur_layer_index}.")
            break
        next_layer = []
        for node in cur_layer:
            for around_node in get_around(node):
                nodes_degree[around_node] -= 1
                if nodes_degree[around_node] == 0:
                    next_layer.append(around_node)
        topo_layers.append(next_layer)
        cur_layer = next_layer
        cur_layer_index += 1

    return topo_layers


# 根据拓扑分层计算节点坐标
def _calc_nodes_pos(
    topo_node_layers: list[list[Node]],
    anchor_pos: NodePosition,
    x_increment: int,
    y_increment: int
) -> dict[Node, NodePosition]:
    nodes_pos: dict[Node, NodePosition] = {}
    for layer_id in range(len(topo_node_layers)):
        for layer_inner_id in range(len(topo_node_layers[layer_id])):
            node = topo_node_layers[layer_id][layer_inner_id]
            nodes_pos[node] = NodePosition(0, 0, 0, 0)
            nodes_pos[node].rel_x = layer_inner_id * x_increment + anchor_pos.rel_x
            nodes_pos[node].rel_y = layer_id * y_increment + anchor_pos.rel_x
            nodes_pos[node].abs_x = layer_inner_id
            nodes_pos[node].abs_y = layer_id

    return nodes_pos


def _remove_virtual_node_layer(anchor: Node, topo_layers: list[list[Node]]) -> list[list[Node]]:
    if anchor.get_operator_type() != VIRTUAL_NODE_OPERATOR_TYPE:
        return topo_layers

    del topo_layers[0]

    return topo_layers


def get_anchor_around_nodes_pos(
    di_graph: nx.DiGraph,
    anchor: Node,
    anchor_pos: NodePosition,
    down_or_up: bool,
    x_increment: int,
    y_increment: int
) -> tuple[
    list[list[Node]],
    dict[Node, NodePosition]
]:
    """
    down_or_up为True表示向下生长，为False表示向上生长
    x_increment为节点横向坐标增量，正数表示节点向右排布，负数表示节点向左排布
    y_increment为节点纵向坐标增量，正数表示节点向下排布，负数表示节点向上排布
    """
    sub_graph = _get_anchor_around_sub_graph(di_graph, anchor, down_or_up)
    topo_layers = _get_topo_layers(sub_graph, down_or_up)
    nodes_pos = _calc_nodes_pos(topo_layers, anchor_pos, x_increment, y_increment)
    topo_layers = _remove_virtual_node_layer(anchor, topo_layers)

    return topo_layers, nodes_pos


class Anchor:
    def __init__(self, anchor: Node):
        self._anchor = anchor
        self._match_nodes: set[Node] = set()
        self._mismatch_nodes: set[Node] = set()

    def set_anchor(self, anchor: Node):
        self._anchor = anchor

    def get_anchor(self) -> Node:
        return self._anchor

    def add_match_nodes(self, match_nodes: set[Node]):
        self._match_nodes.update(match_nodes)

    def add_mismatch_nodes(self, mismatch_nodes: set[Node]):
        self._mismatch_nodes.update(mismatch_nodes)

    def clear(self):
        self._match_nodes.clear()
        self._mismatch_nodes.clear()


class AnchorManager:
    def __init__(self, executing_order: ExecutingOrder):
        self._executing_order = executing_order
        self._first_level_anchor = Anchor(self._executing_order.get_node_manager().get_first_node())
        self._topo_layers, self._nodes_pos = get_anchor_around_nodes_pos(
            self._executing_order.get_graph(),
            self._first_level_anchor.get_anchor(),
            NodePosition(0, 0, 0, 0),
            False,
            1,
            -1
        )
        self._second_level_anchors: list[Anchor] = []

    def get_first_level_anchor(self):
        return self._first_level_anchor

    def get_topo_layers(self):
        return self._topo_layers

    def get_node_pos(self, node: Node) -> NodePosition:
        return self._nodes_pos[node]

    def get_second_level_anchor(self, anchor_index: int):
        return self._second_level_anchors[anchor_index]

    def get_all_anchors(self) -> set[Node]:
        all_anchors = set()
        all_anchors.add(self._first_level_anchor.get_anchor())
        for second_level_anchor in self._second_level_anchors:
            all_anchors.add(second_level_anchor.get_anchor())
        return all_anchors

    def update_first_level_anchor(self, anchor: Node):
        self._first_level_anchor = Anchor(anchor)
        self._topo_layers, self._nodes_pos = get_anchor_around_nodes_pos(
            self._executing_order.get_graph(),
            self._first_level_anchor.get_anchor(),
            NodePosition(0, 0, 0, 0),
            False,
            1,
            -1
        )
        self._second_level_anchors.clear()

    def add_second_level_anchor(self, anchor: Node):
        self._second_level_anchors.append(Anchor(anchor))

    def clear(self):
        self._first_level_anchor.clear()
        self._second_level_anchors.clear()
