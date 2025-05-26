import logging
import networkx as nx
from ExecutingOrder.ExecutingOrder import ExecutingOrder
from ExecutingOrder.NodeManager import SINK_VIRTUAL_NODE_OPERATOR_TYPE, Node
from ColorManager import ColorManager, NodeColor
from DiGraphComparator.DiGraphComparatorTopoLayers import DiGraphComparatorTopoLayers


class NodePosition:
    def __init__(self, abs_x: int, abs_y: int, rel_x: int, rel_y: int):
        self.abs_x = abs_x
        self.abs_y = abs_y
        self.rel_x = rel_x
        self.rel_y = rel_y


# 根据anchor获取周边子图
def _get_anchor_around_sub_graph(graph: nx.DiGraph, anchor: Node, up_direction: bool) -> nx.DiGraph:
    get_around = graph.successors
    if up_direction:
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


def _get_nodes_degree(graph: nx.DiGraph, up_direction: bool) -> dict[Node, int]:
    nodes_degree = {node: 0 for node in graph.nodes()}
    if up_direction:
        for src, _ in graph.edges():
            nodes_degree[src] += 1
    else:
        for _, dst in graph.edges():
            nodes_degree[dst] += 1
    return nodes_degree


def _get_topo_layers(graph: nx.DiGraph, up_direction: bool) -> list[list[Node]]:
    nodes_degree = _get_nodes_degree(graph, up_direction)

    get_around = graph.successors
    if up_direction:
        get_around = graph.predecessors

    cur_layer = [node for node, degree in nodes_degree.items() if degree == 0]
    cur_layer_index = 0
    topo_layers = [cur_layer]
    while True:
        next_layer = []
        for node in cur_layer:
            for around_node in get_around(node):
                nodes_degree[around_node] -= 1
                if nodes_degree[around_node] == 0:
                    next_layer.append(around_node)
        if len(next_layer) == 0:
            logging.info(f"Max layer num is {cur_layer_index + 1}.")
            break
        topo_layers.append(next_layer)
        cur_layer = next_layer
        cur_layer_index += 1

    return topo_layers


def get_anchor_around_nodes_topo_layers(
        di_graph: nx.DiGraph,
        anchor: Node,
        up_direction: bool = True
) -> list[list[Node]]:
    """
    up_direction为True表示向上生长，为False表示向下生长
    """
    sub_graph = _get_anchor_around_sub_graph(di_graph, anchor, up_direction)
    topo_layers = _get_topo_layers(sub_graph, up_direction)

    return topo_layers


def get_anchor_around_nodes_pos(
        topo_layers: list[list[Node]],
        anchor_pos: NodePosition,
        x_increment: int,
        y_increment: int
) -> dict[Node, NodePosition]:
    """
    x_increment为节点横向坐标增量，正数表示节点向右排布，负数表示节点向左排布
    y_increment为节点纵向坐标增量，正数表示节点向下排布，负数表示节点向上排布
    """
    nodes_pos: dict[Node, NodePosition] = {}
    for layer_id in range(len(topo_layers)):
        for layer_inner_id in range(len(topo_layers[layer_id])):
            node = topo_layers[layer_id][layer_inner_id]
            abs_x = layer_inner_id
            abs_y = layer_id
            rel_x = layer_inner_id * x_increment + anchor_pos.rel_x
            rel_y = layer_id * y_increment + anchor_pos.rel_x
            nodes_pos[node] = NodePosition(abs_x, abs_y, rel_x, rel_y)

    return nodes_pos


class AnchorManager:
    def __init__(self, src_executing_order: ExecutingOrder, dst_executing_order: ExecutingOrder):
        self._src_graph = src_executing_order.get_graph()
        self._dst_graph = dst_executing_order.get_graph()

        self._first_level_anchor_comparator = DiGraphComparatorTopoLayers(
            self._src_graph,
            self._dst_graph,
            src_executing_order.get_node_manager().get_first_node(),
            dst_executing_order.get_node_manager().get_first_node()
        )
        """
        使用一级锚点计算节点位置，用于前端展示
        """
        self._src_topo_layers = get_anchor_around_nodes_topo_layers(
            self._src_graph,
            self.get_src_first_level_anchor()
        )
        self._dst_topo_layers = get_anchor_around_nodes_topo_layers(
            self._dst_graph,
            self.get_dst_first_level_anchor()
        )
        self._src_nodes_pos = get_anchor_around_nodes_pos(
            self._src_topo_layers,
            NodePosition(0, 0, 0, 0),
            1,
            -1
        )
        self._dst_nodes_pos = get_anchor_around_nodes_pos(
            self._dst_topo_layers,
            NodePosition(0, 0, 0, 0),
            1,
            -1
        )

        self._second_level_anchors_comparator: list[DiGraphComparatorTopoLayers] = []

    def get_first_level_anchor_comparator(self) -> DiGraphComparatorTopoLayers:
        return self._first_level_anchor_comparator

    def get_src_topo_layers(self) -> list[list[Node]]:
        return self._src_topo_layers

    def get_dst_topo_layers(self) -> list[list[Node]]:
        return self._dst_topo_layers

    def get_src_node_pos(self, node: Node) -> NodePosition:
        return self._src_nodes_pos[node]

    def get_dst_node_pos(self, node: Node) -> NodePosition:
        return self._dst_nodes_pos[node]

    def get_second_level_anchors_comparator(self) -> list[DiGraphComparatorTopoLayers]:
        return self._second_level_anchors_comparator

    def get_src_first_level_anchor(self) -> Node:
        return self._first_level_anchor_comparator.get_src_anchor()

    def get_dst_first_level_anchor(self) -> Node:
        return self._first_level_anchor_comparator.get_dst_anchor()

    def get_src_all_anchors(self) -> set[Node]:
        all_anchors = set()
        all_anchors.add(self.get_src_first_level_anchor())
        for comparator in self._second_level_anchors_comparator:
            all_anchors.add(comparator.get_src_anchor())
        return all_anchors

    def get_dst_all_anchors(self) -> set[Node]:
        all_anchors = set()
        all_anchors.add(self.get_dst_first_level_anchor())
        for comparator in self._second_level_anchors_comparator:
            all_anchors.add(comparator.get_dst_anchor())
        return all_anchors

    def get_src_node_color(self, node: Node) -> NodeColor:
        if node in self.get_src_all_anchors():
            return ColorManager.get_anchor_node_color()
        for comparator in [*self._second_level_anchors_comparator, self._first_level_anchor_comparator]:
            if comparator.is_compared():
                if comparator.src_node_is_matched(node):
                    return ColorManager.get_matched_node_color()
                if comparator.src_node_is_mismatched(node):
                    return ColorManager.get_mismatched_node_color()
        return ColorManager.get_default_node_color()

    def get_dst_node_color(self, node: Node) -> NodeColor:
        if node in self.get_dst_all_anchors():
            return ColorManager.get_anchor_node_color()
        for comparator in [*self._second_level_anchors_comparator, self._first_level_anchor_comparator]:
            if comparator.is_compared():
                if comparator.dst_node_is_matched(node):
                    return ColorManager.get_matched_node_color()
                if comparator.dst_node_is_mismatched(node):
                    return ColorManager.get_mismatched_node_color()
        return ColorManager.get_default_node_color()

    def get_src_edge_color(self, src_node: Node, dst_node: Node) -> str:
        return ColorManager.get_edge_color(src_node, dst_node)

    def get_dst_edge_color(self, src_node: Node, dst_node: Node) -> str:
        return ColorManager.get_edge_color(src_node, dst_node)

    def update_src_first_level_anchor(self, anchor: Node):
        self._first_level_anchor_comparator.clear_compared_result()
        self._first_level_anchor_comparator.set_src_anchor(anchor)
        self._src_topo_layers = get_anchor_around_nodes_topo_layers(
            self._src_graph,
            self.get_src_first_level_anchor()
        )
        self._src_nodes_pos = get_anchor_around_nodes_pos(
            self._src_topo_layers,
            NodePosition(0, 0, 0, 0),
            1,
            -1
        )

    def update_dst_first_level_anchor(self, anchor: Node):
        self._first_level_anchor_comparator.clear_compared_result()
        self._first_level_anchor_comparator.set_dst_anchor(anchor)
        self._dst_topo_layers = get_anchor_around_nodes_topo_layers(
            self._dst_graph,
            self.get_dst_first_level_anchor()
        )
        self._dst_nodes_pos = get_anchor_around_nodes_pos(
            self._dst_topo_layers,
            NodePosition(0, 0, 0, 0),
            1,
            -1
        )

    def add_second_level_anchor(self, src_anchor: Node, dst_anchor: Node):
        self._second_level_anchors_comparator.append(
            DiGraphComparatorTopoLayers(
                self._src_graph,
                self._dst_graph,
                src_anchor,
                dst_anchor
            )
        )

    def delete_second_level_anchor(self, src_nodes_id: list[int], dst_nodes_id: list[int]):
        remove_ids_set = set()
        for comparator_id in range(len(self._second_level_anchors_comparator)):
            comparator = self._second_level_anchors_comparator[comparator_id]
            if comparator.get_src_anchor().get_node_id() in src_nodes_id or \
                    comparator.get_dst_anchor().get_node_id() in dst_nodes_id:
                remove_ids_set.add(comparator_id)
        remove_ids = list(remove_ids_set)
        remove_ids.sort(reverse=True)
        for remove_id in remove_ids:
            del self._second_level_anchors_comparator[remove_id]

    def clear_second_level_anchors(self):
        self._second_level_anchors_comparator.clear()

    def clear_first_level_anchor_compared_result(self):
        self._first_level_anchor_comparator.clear_compared_result()

    def clear_second_level_anchors_compared_result(self):
        for comparator in self._second_level_anchors_comparator:
            comparator.clear_compared_result()

    def first_level_anchor_compare(self, up_direction: bool):
        self._first_level_anchor_comparator.compare_graphs(
            up_direction,
            self._src_topo_layers[1:],  # 去掉源锚点所在的一层，锚点认为是比配的，不需要比较
            self._dst_topo_layers[1:]  # 去掉源锚点所在的一层，锚点认为是比配的，不需要比较
        )

    def second_level_anchors_compare(self, up_direction: bool):
        for comparator in self._second_level_anchors_comparator:
            src_topo_layers = get_anchor_around_nodes_topo_layers(
                self._src_graph,
                comparator.get_src_anchor(),
                up_direction
            )
            dst_topo_layers = get_anchor_around_nodes_topo_layers(
                self._dst_graph,
                comparator.get_dst_anchor(),
                up_direction
            )
            if not up_direction:
                src_topo_layers = src_topo_layers[-1:0:-1]  # 向下生长时，索引值小的为顶层，需要反转，并去掉源锚点所在的一层
                dst_topo_layers = dst_topo_layers[-1:0:-1]  # 向下生长时，索引值小的为顶层，需要反转，并去掉源锚点所在的一层
            else:
                src_topo_layers = src_topo_layers[1:]  # 去掉源锚点所在的一层，锚点认为是比配的，不需要比较
                dst_topo_layers = dst_topo_layers[1:]  # 去掉源锚点所在的一层，锚点认为是比配的，不需要比较
            comparator.compare_graphs(
                up_direction,
                src_topo_layers,
                dst_topo_layers
            )

    @staticmethod
    def _get_mismatch_node_mismatch_source_info(anchor: Node, padding: str) -> list[str]:
        info_lines: list[str] = []
        if anchor.get_operator_type() == SINK_VIRTUAL_NODE_OPERATOR_TYPE:
            info_lines.append(padding + "基于整图找到的差异")
        else:
            info_lines.append(padding +
                              "基于锚点" +
                              f" 算子类型 {anchor.get_operator_type()} 行号 {anchor.get_line_num()} " +
                              f"找到的差异"
                              )
        return info_lines

    def get_src_node_mismatch_info(self, node: Node, padding: str) -> list[str]:
        info_lines: list[str] = []
        if node in self.get_src_all_anchors():
            return info_lines
        for comparator in [*self._second_level_anchors_comparator, self._first_level_anchor_comparator]:
            if comparator.is_compared():
                if comparator.src_node_is_mismatched(node):
                    anchor = comparator.get_src_anchor()
                    info_lines.extend(AnchorManager._get_mismatch_node_mismatch_source_info(anchor, ""))
                    info_lines.extend(comparator.get_src_mismatch_node_info(node, padding + " " * 2))
                    return info_lines
        return info_lines

    def get_dst_node_mismatch_info(self, node: Node, padding: str) -> list[str]:
        info_lines: list[str] = []
        if node in self.get_dst_all_anchors():
            return info_lines
        for comparator in [*self._second_level_anchors_comparator, self._first_level_anchor_comparator]:
            if comparator.is_compared():
                if comparator.dst_node_is_mismatched(node):
                    anchor = comparator.get_dst_anchor()
                    info_lines.extend(AnchorManager._get_mismatch_node_mismatch_source_info(anchor, ""))
                    info_lines.extend(comparator.get_dst_mismatch_node_info(node, padding + " " * 2))
                    return info_lines
        return info_lines
