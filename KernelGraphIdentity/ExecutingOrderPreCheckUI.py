import logging
import os
import sys
import networkx as nx
from ColorManager import SELECTED_EDGE_COLOR, SELECTED_NODE_BACKGROUND_COLOR, SELECTED_NODE_BORDER_COLOR
from pyvis.network import Network
from AnchorManager import AnchorManager
from ExecutingOrder.ExecutingOrder import ExecutingOrder
from ExecutingOrder.NodeManager import SINK_VIRTUAL_NODE_OPERATOR_TYPE, SOURCE_VIRTUAL_NODE_OPERATOR_TYPE
from ExecutingOrder.NodeManager import Node
from DiGraphComparator.DiGraphComparatorIsolatedNodes import DiGraphComparatorIsolatedNodes

DEFAULT_CANVAS_HEIGHT = 600
DEFAULT_CANVAS_WIDTH = 800
CANVAS_NODE_X_INTERVAL = 200
CANVAS_NODE_Y_INTERVAL = 100
CANVAS_FONT_SIZE = 10


def _gen_pyvis_network(canvas_height: int, canvas_width: int) -> Network:
    height = f"{canvas_height}px"
    width = f"{canvas_width}px"

    network = Network(
        directed=True,
        height=height,
        width=width,
        notebook=False,
        cdn_resources="remote"
    )

    options = """
    {
        "physics": {
            "enabled": false
        },
        "interaction": {
            "multiselect": true
        },
        "edges": {
            "smooth": {
                "type": "discrete"
            }
        }
    }
    """
    network.set_options(options)

    return network


def _add_nodes(
        show_graph: nx.DiGraph,
        network: Network,
        get_node_pos,
        get_node_color
):
    node_list = []
    label_list = []
    size_list = []
    x_list = []
    y_list = []
    color_list = []
    for node in show_graph.nodes():
        node_position = get_node_pos(node)
        x = node_position.rel_x * CANVAS_NODE_X_INTERVAL
        y = node_position.rel_y * CANVAS_NODE_Y_INTERVAL
        label = f"{node.get_operator_type()} {node.get_line_num()} {node_position.abs_y}"
        node_color = get_node_color(node)

        node_list.append(node.get_node_id())
        label_list.append(label)
        size_list.append(CANVAS_FONT_SIZE)
        x_list.append(x)
        y_list.append(y)
        color_list.append({
            "background": node_color.background_color,
            "border": node_color.border_color,
            "highlight": {
                "background": SELECTED_NODE_BACKGROUND_COLOR,
                "border": SELECTED_NODE_BORDER_COLOR
            }
        })

    network.add_nodes(node_list, label=label_list, size=size_list, x=x_list, y=y_list, color=color_list)


def _add_edges(show_graph: nx.DiGraph, executing_order: ExecutingOrder, get_edge_color, network: Network):
    for src_node, dst_node in show_graph.edges():
        title = str(executing_order.get_edge_manager().get_edge(src_node, dst_node))
        edge_color = {
            "color": get_edge_color(src_node, dst_node),
            "highlight": SELECTED_EDGE_COLOR
        }

        network.add_edge(
            src_node.get_node_id(),
            dst_node.get_node_id(),
            title=title,
            color=edge_color
        )


def _save_html(
        show_graph: nx.DiGraph,
        executing_order: ExecutingOrder,
        get_node_pos,
        get_node_color,
        get_edge_color,
        html_path: str,
        canvas_height: int,
        canvas_width: int
):
    network = _gen_pyvis_network(canvas_height, canvas_width)
    _add_nodes(show_graph, network, get_node_pos, get_node_color)
    _add_edges(show_graph, executing_order, get_edge_color, network)
    network.save_graph(html_path)


def _remove_virtual_node_layer(topo_layers: list[list[Node]]) -> list[list[Node]]:
    remove_layers = []
    for layer_id in range(len(topo_layers)):
        topo_layer = topo_layers[layer_id]
        operator_type = topo_layer[0].get_operator_type()
        if (operator_type == SOURCE_VIRTUAL_NODE_OPERATOR_TYPE or
                operator_type == SINK_VIRTUAL_NODE_OPERATOR_TYPE):
            remove_layers.append(layer_id)

    for layer_id in remove_layers[::-1]:
        del topo_layers[layer_id]

    return topo_layers


def _get_show_graph(graph: nx.DiGraph, topo_layers: list[list[Node]], show_layer_num: int, show_all_layer: bool):
    topo_layers = topo_layers.copy()
    _remove_virtual_node_layer(topo_layers)

    show_graph_nodes = set()
    show_layer_num = min(show_layer_num, len(topo_layers))
    if show_all_layer:
        show_layer_num = len(topo_layers)
    for layer_id in range(show_layer_num):
        show_graph_nodes.update(topo_layers[layer_id])
    return graph.subgraph(show_graph_nodes)


class ExecutingOrderPreCheckUI:
    def __init__(self):
        self._set_path()
        self._set_logging()

        # 执行进度
        self._progress = {"current": 0, "total": 100, "message": "处理中..."}

        # 初始化画布大小
        self._canvas_height = DEFAULT_CANVAS_HEIGHT
        self._canvas_width = DEFAULT_CANVAS_WIDTH

        # 初始化执行序
        self._src_executing_order: ExecutingOrder = ExecutingOrder(self._init_executing_order_file_path)
        self._dst_executing_order: ExecutingOrder = ExecutingOrder(self._init_executing_order_file_path)

        # 初始化锚点
        self._anchor_manager: AnchorManager = AnchorManager(self._src_executing_order, self._dst_executing_order)

        # 初始化显示拓扑层数
        self._show_layer_num = 10
        # 整图模式不受拓扑层数影响，显示所有层
        self._show_all_layer = True

        # 初始化显示的图
        self._set_src_show_graph()
        self._set_dst_show_graph()

        self.save_src_html()
        self.save_dst_html()

    def _set_path(self):
        self._bash_path = os.path.abspath(".")
        if getattr(sys, 'frozen', False):
            self._bash_path = sys._MEIPASS  # PyInstaller临时解压目录
        self._init_executing_order_file_path = os.path.join(self._bash_path, "static", "init.txt")
        self._src_executing_order_file_path = os.path.join(self._bash_path, "static", "src.ir")
        self._dst_executing_order_file_path = os.path.join(self._bash_path, "static", "dst.ir")
        self._src_html_path = os.path.join(self._bash_path, "static", "src_graph.html")
        self._dst_html_path = os.path.join(self._bash_path, "static", "dst_graph.html")

        self._log_path = "app.log"
        if getattr(sys, 'frozen', False):
            self._log_path = os.path.join(os.path.dirname(sys.executable), "app.log")

    def _set_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(self._log_path),
                logging.StreamHandler()
            ]
        )

    def get_src_executing_order_file_path(self) -> str:
        return self._src_executing_order_file_path

    def get_dst_executing_order_file_path(self) -> str:
        return self._dst_executing_order_file_path

    def get_src_executing_order(self) -> ExecutingOrder:
        return self._src_executing_order

    def get_dst_executing_order(self) -> ExecutingOrder:
        return self._dst_executing_order

    def get_anchor_manager(self) -> AnchorManager:
        return self._anchor_manager

    def get_progress(self) -> dict:
        return self._progress

    def set_canvas_size(self, canvas_height: int, canvas_width: int):
        self._canvas_height = canvas_height
        self._canvas_width = canvas_width

        self.save_src_html()
        self.save_dst_html()

    def set_show_all_layer(self, show_all_layer: bool):
        self._show_all_layer = show_all_layer

    def get_show_all_layer(self) -> bool:
        return self._show_all_layer

    def _set_src_show_graph(self):
        self._src_show_graph = _get_show_graph(
            self._src_executing_order.get_graph(),
            self._anchor_manager.get_src_topo_layers(),
            self._show_layer_num,
            self._show_all_layer
        )

    def _set_dst_show_graph(self):
        self._dst_show_graph = _get_show_graph(
            self._dst_executing_order.get_graph(),
            self._anchor_manager.get_dst_topo_layers(),
            self._show_layer_num,
            self._show_all_layer
        )

    def save_src_html(self):
        _save_html(
            self._src_show_graph,
            self._src_executing_order,
            self._anchor_manager.get_src_node_pos,
            self._anchor_manager.get_src_node_color,
            self._anchor_manager.get_src_edge_color,
            self._src_html_path,
            self._canvas_height,
            self._canvas_width
        )

    def save_dst_html(self):
        _save_html(
            self._dst_show_graph,
            self._dst_executing_order,
            self._anchor_manager.get_dst_node_pos,
            self._anchor_manager.get_dst_node_color,
            self._anchor_manager.get_dst_edge_color,
            self._dst_html_path,
            self._canvas_height,
            self._canvas_width
        )

    def reload_src_executing_order(self):
        self._src_executing_order: ExecutingOrder = ExecutingOrder(self._src_executing_order_file_path)
        self._anchor_manager: AnchorManager = AnchorManager(self._src_executing_order, self._dst_executing_order)

        self._set_src_show_graph()
        self._set_dst_show_graph()

        self.save_src_html()
        self.save_dst_html()

    def reload_dst_executing_order(self):
        self._dst_executing_order: ExecutingOrder = ExecutingOrder(self._dst_executing_order_file_path)
        self._anchor_manager: AnchorManager = AnchorManager(self._src_executing_order, self._dst_executing_order)

        self._set_src_show_graph()
        self._set_dst_show_graph()

        self.save_src_html()
        self.save_dst_html()

    def get_src_node_by_line_num(self, line_num: int) -> Node:
        return self._src_executing_order.get_node_manager().get_node_by_line_num(line_num)

    def get_dst_node_by_line_num(self, line_num: int) -> Node:
        return self._dst_executing_order.get_node_manager().get_node_by_line_num(line_num)

    def get_src_node_by_node_id(self, node_id: int) -> Node:
        return self._src_executing_order.get_node_manager().get_node_by_node_id(node_id)

    def get_dst_node_by_node_id(self, node_id: int) -> Node:
        return self._dst_executing_order.get_node_manager().get_node_by_node_id(node_id)

    def get_src_sink_virtual_node(self) -> Node:
        return self._src_executing_order.get_node_manager().get_sink_virtual_node()

    def get_dst_sink_virtual_node(self) -> Node:
        return self._dst_executing_order.get_node_manager().get_sink_virtual_node()

    def get_src_node_mismatch_info(self, node: Node, padding: str) -> list[str]:
        return self._anchor_manager.get_src_node_mismatch_info(node, padding)

    def get_dst_node_mismatch_info(self, node: Node, padding: str) -> list[str]:
        return self._anchor_manager.get_dst_node_mismatch_info(node, padding)

    def update_src_first_level_anchor(self, anchor: Node):
        self._anchor_manager.update_src_first_level_anchor(anchor)
        self._anchor_manager.clear_second_level_anchors()

        self._set_src_show_graph()

        self.save_src_html()
        self.save_dst_html()

    def update_dst_first_level_anchor(self, anchor: Node):
        self._anchor_manager.update_dst_first_level_anchor(anchor)
        self._anchor_manager.clear_second_level_anchors()

        self._set_dst_show_graph()

        self.save_src_html()
        self.save_dst_html()

    def fuse_src_nodes_cycle_check(self, fuse_nodes_id: list[int]):
        return self._src_executing_order.fuse_nodes_cycle_check(fuse_nodes_id)

    def fuse_dst_nodes_cycle_check(self, fuse_nodes_id: list[int]):
        return self._dst_executing_order.fuse_nodes_cycle_check(fuse_nodes_id)

    def fuse_src_nodes(self, fuse_nodes_id: list[int], peer_node: Node, fused_node_operator_type: str):
        fused_node = self._src_executing_order.fuse_nodes(fuse_nodes_id, peer_node, fused_node_operator_type)

        # 融合节点包含二级锚点时，删除该二级锚点
        self._anchor_manager.delete_second_level_anchor(fuse_nodes_id, [])

        if self._anchor_manager.get_src_first_level_anchor().get_node_id() in fuse_nodes_id:
            # 融合节点包含一级锚点时，需要更新一级锚点为融合后节点
            self._anchor_manager.update_src_first_level_anchor(fused_node)
        else:
            # 融合后拓扑结构发生变化，通过重新设置一级锚点，重新计算拓扑层和节点位置
            self._anchor_manager.update_src_first_level_anchor(self._anchor_manager.get_src_first_level_anchor())

        self._anchor_manager.clear_second_level_anchors_compared_result()

        self._set_src_show_graph()

        self.save_src_html()
        self.save_dst_html()

        return fused_node.get_node_id()

    def fuse_dst_nodes(self, fuse_nodes_id: list[int], peer_node: Node, fused_node_operator_type: str):
        fused_node = self._dst_executing_order.fuse_nodes(fuse_nodes_id, peer_node, fused_node_operator_type)

        # 融合节点包含二级锚点时，删除该二级锚点
        self._anchor_manager.delete_second_level_anchor([], fuse_nodes_id)

        if self._anchor_manager.get_dst_first_level_anchor().get_node_id() in fuse_nodes_id:
            # 融合节点包含一级锚点时，需要更新一级锚点为融合后节点
            self._anchor_manager.update_dst_first_level_anchor(fused_node)
        else:
            # 融合后拓扑结构发生变化，通过重新设置一级锚点，重新计算拓扑层和节点位置
            self._anchor_manager.update_dst_first_level_anchor(self._anchor_manager.get_dst_first_level_anchor())

        self._anchor_manager.clear_second_level_anchors_compared_result()

        self._set_dst_show_graph()

        self.save_src_html()
        self.save_dst_html()

        return fused_node.get_node_id()

    def delete_src_nodes(self, nodes_id: list[int]):
        self._src_executing_order.delete_nodes(nodes_id)

        # 删除节点后拓扑结构发生变化，通过重新设置一级锚点，重新计算拓扑层和节点位置
        self._anchor_manager.update_src_first_level_anchor(self._anchor_manager.get_src_first_level_anchor())
        self._anchor_manager.clear_second_level_anchors()

        self._set_src_show_graph()

        self.save_src_html()
        self.save_dst_html()

    def delete_dst_nodes(self, nodes_id: list[int]):
        self._dst_executing_order.delete_nodes(nodes_id)

        # 删除节点后拓扑结构发生变化，通过重新设置一级锚点，重新计算拓扑层和节点位置
        self._anchor_manager.update_dst_first_level_anchor(self._anchor_manager.get_dst_first_level_anchor())
        self._anchor_manager.clear_second_level_anchors()

        self._set_dst_show_graph()

        self.save_src_html()
        self.save_dst_html()

    def delete_src_edges(self, edges: list[dict[str, int]]):
        self._src_executing_order.delete_edges(edges)

        # 删除边后拓扑结构发生变化，通过重新设置一级锚点，重新计算拓扑层和节点位置
        self._anchor_manager.update_src_first_level_anchor(self._anchor_manager.get_src_first_level_anchor())
        self._anchor_manager.clear_second_level_anchors()

        self._set_src_show_graph()

        self.save_src_html()
        self.save_dst_html()

    def delete_dst_edges(self, edges: list[dict[str, int]]):
        self._dst_executing_order.delete_edges(edges)

        # 删除边后拓扑结构发生变化，通过重新设置一级锚点，重新计算拓扑层和节点位置
        self._anchor_manager.update_dst_first_level_anchor(self._anchor_manager.get_dst_first_level_anchor())
        self._anchor_manager.clear_second_level_anchors()

        self._set_dst_show_graph()

        self.save_src_html()
        self.save_dst_html()

    def set_layer_num(self, layer_num: int):
        self._show_layer_num = layer_num

        self._set_src_show_graph()
        self._set_dst_show_graph()

        self.save_src_html()
        self.save_dst_html()

    def add_second_level_anchor(self, src_anchor: Node, dst_anchor: Node):
        self._anchor_manager.add_second_level_anchor(src_anchor, dst_anchor)

        self.save_src_html()
        self.save_dst_html()

    def delete_second_level_anchor(self, src_anchors_id: list[int], dst_anchors_id: list[int]):
        self._anchor_manager.delete_second_level_anchor(src_anchors_id, dst_anchors_id)

        self.save_src_html()
        self.save_dst_html()

    def _compare_before(self, up_direction: bool):
        """
        已比较过，但当前要比较的方向和之前不一样，需要清除原来比较结果，重新比较
        """
        if not self._anchor_manager.get_first_level_anchor_comparator().is_compared():
            return
        if self._anchor_manager.get_first_level_anchor_comparator().get_up_direction() == up_direction:
            return
        self._anchor_manager.clear_first_level_anchor_compared_result()
        self._anchor_manager.clear_second_level_anchors()
        return

    def up_direction_compare(self):
        self._compare_before(True)
        self._anchor_manager.first_level_anchor_compare(True)
        self._anchor_manager.second_level_anchors_compare(True)

        self.save_src_html()
        self.save_dst_html()

    def down_direction_compare(self):
        self._compare_before(False)
        self._anchor_manager.first_level_anchor_compare(False)
        self._anchor_manager.second_level_anchors_compare(False)

        self.save_src_html()
        self.save_dst_html()

    def compare_isolated_nodes(self) -> list[str]:
        comparator = DiGraphComparatorIsolatedNodes(
            self._src_executing_order.get_graph(),
            self._dst_executing_order.get_graph()
        )
        info_lines = comparator.compare_graphs(
            self._src_executing_order.get_node_manager().get_isolated_nodes(),
            self._dst_executing_order.get_node_manager().get_isolated_nodes()
        )
        return info_lines
