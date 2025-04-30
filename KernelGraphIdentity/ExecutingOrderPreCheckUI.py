import logging
import os
import sys
import networkx as nx
from ColorManager import SELECTED_NODE_BACKGROUND_COLOR, SELECTED_NODE_BORDER_COLOR
from pyvis.network import Network
from AnchorManager import AnchorManager, NodePosition, get_anchor_around_nodes_pos
from ColorManager import ColorManager
from ExecutingOrder.ExecutingOrder import ExecutingOrder
from ExecutingOrder.NodeManager import Node
from GraphComparator.GraphComparatorAnchor import GraphComparatorAnchor


DEFAULT_CANVAS_HEIGHT = 600
DEFAULT_CANVAS_WIDTH = 800


def _set_logging():
    log_path = "app.log"
    if getattr(sys, 'frozen', False):
        log_path = os.path.join(os.path.dirname(sys.executable), "app.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


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
        }
    }
    """
    network.set_options(options)

    return network


def _add_nodes(
    show_graph: nx.DiGraph,
    network: Network,
    anchor_manager: AnchorManager,
    color_manager: ColorManager
):
    for node in show_graph.nodes():
        node_position = anchor_manager.get_node_pos(node)
        x = node_position.rel_x * 200
        y = node_position.rel_y * 100
        label = f"{node.get_operator_type()} {node.get_line_num()} {node_position.abs_y}"
        node_color = color_manager.get_node_color(node)

        network.add_node(
            node.get_node_id(),
            label=label,
            size=10,
            x=x,
            y=y,
            color={
                "background": node_color.background_color,
                "border": node_color.border_color,
                "highlight": {
                    "background": SELECTED_NODE_BACKGROUND_COLOR,
                    "border": SELECTED_NODE_BORDER_COLOR
                }
            }
        )


def _add_edges(show_graph: nx.DiGraph, executing_order: ExecutingOrder, color_manager: ColorManager, network: Network):
    for src_node, dst_node in show_graph.edges():
        title = str(executing_order.get_edge_manager().get_edge(src_node, dst_node))
        edge_color = color_manager.get_edge_color(src_node, dst_node)

        network.add_edge(src_node.get_node_id(), dst_node.get_node_id(), title=title, color=edge_color)


def _save_html(
    show_graph: nx.DiGraph,
    executing_order: ExecutingOrder,
    anchor_manager: AnchorManager,
    color_manager: ColorManager,
    html_path: str,
    canvas_height: int,
    canvas_width: int
):
    network = _gen_pyvis_network(canvas_height, canvas_width)
    _add_nodes(show_graph, network, anchor_manager, color_manager)
    _add_edges(show_graph, executing_order, color_manager, network)
    network.save_graph(html_path)


def _get_show_graph(graph: nx.DiGraph, topo_layers: list[list[Node]], show_layer_num: int, show_all_layer: bool):
    show_graph_nodes = set()
    show_layer_num = min(show_layer_num, len(topo_layers))
    if show_all_layer:
        show_layer_num = len(topo_layers)
    for layer_id in range(show_layer_num):
        show_graph_nodes.update(topo_layers[layer_id])
    return graph.subgraph(show_graph_nodes)


class ExecutingOrderPreCheckUI:
    def __init__(self):
        _set_logging()
        self._set_path()

        # 初始化画布大小
        self._canvas_height = DEFAULT_CANVAS_HEIGHT
        self._canvas_width = DEFAULT_CANVAS_WIDTH

        # 初始化执行序
        self._src_executing_order: ExecutingOrder = ExecutingOrder(self._init_executing_order_file_path)
        self._dst_executing_order: ExecutingOrder = ExecutingOrder(self._init_executing_order_file_path)

        # 初始化锚点
        self._src_anchor_manager: AnchorManager = AnchorManager(self._src_executing_order)
        self._dst_anchor_manager: AnchorManager = AnchorManager(self._dst_executing_order)
        self._is_first_level_anchor_compared = False
        self._is_second_level_anchor_compared = []

        # 初始化颜色
        self._src_color_manager: ColorManager = ColorManager()
        self._src_color_manager.set_anchors_color({self._src_anchor_manager.get_first_level_anchor().get_anchor()})
        self._dst_color_manager: ColorManager = ColorManager()
        self._dst_color_manager.set_anchors_color({self._dst_anchor_manager.get_first_level_anchor().get_anchor()})

        # 初始化显示拓扑层数
        self._show_layer_num = 10
        # 整图模式不受拓扑层数影响，显示所有层
        self._show_all_layer = False

        # 初始化显示的图
        self._src_show_graph = _get_show_graph(
            self._src_executing_order.get_graph(),
            self._src_anchor_manager.get_topo_layers(),
            self._show_layer_num,
            self._show_all_layer
        )
        self._dst_show_graph = _get_show_graph(
            self._dst_executing_order.get_graph(),
            self._dst_anchor_manager.get_topo_layers(),
            self._show_layer_num,
            self._show_all_layer
        )

    def _set_path(self):
        self._bash_path = os.path.abspath(".")
        if getattr(sys, 'frozen', False):
            self._bash_path = sys._MEIPASS  # PyInstaller临时解压目录
        self._init_executing_order_file_path = os.path.join(self._bash_path, "static", "init.txt")
        self._src_executing_order_file_path = os.path.join(self._bash_path, "static", "src.ir")
        self._dst_executing_order_file_path = os.path.join(self._bash_path, "static", "dst.ir")
        self._src_html_path = os.path.join(self._bash_path, "static", "src_graph.html")
        self._dst_html_path = os.path.join(self._bash_path, "static", "dst_graph.html")

    def get_src_executing_order_file_path(self) -> str:
        return self._src_executing_order_file_path

    def get_dst_executing_order_file_path(self) -> str:
        return self._dst_executing_order_file_path

    def get_src_executing_order(self) -> ExecutingOrder:
        return self._src_executing_order

    def get_dst_executing_order(self) -> ExecutingOrder:
        return self._dst_executing_order

    def set_canvas_size(self, canvas_height: int, canvas_width: int):
        self._canvas_height = canvas_height
        self._canvas_width = canvas_width

    def set_show_all_layer(self, show_all_layer: bool):
        self._show_all_layer = show_all_layer

    def save_src_html(self):
        _save_html(
            self._src_show_graph,
            self._src_executing_order,
            self._src_anchor_manager,
            self._src_color_manager,
            self._src_html_path,
            self._canvas_height,
            self._canvas_width
        )

    def save_dst_html(self):
        _save_html(
            self._dst_show_graph,
            self._dst_executing_order,
            self._dst_anchor_manager,
            self._dst_color_manager,
            self._dst_html_path,
            self._canvas_height,
            self._canvas_width
        )

    def update_src_executing_order(self):
        self._src_executing_order: ExecutingOrder = ExecutingOrder(self._src_executing_order_file_path)
        self._src_anchor_manager: AnchorManager = AnchorManager(self._src_executing_order)
        self._src_color_manager: ColorManager = ColorManager()
        self._src_color_manager.set_anchors_color({self._src_anchor_manager.get_first_level_anchor().get_anchor()})
        self._src_show_graph = _get_show_graph(
            self._src_executing_order.get_graph(),
            self._src_anchor_manager.get_topo_layers(),
            self._show_layer_num,
            self._show_all_layer
        )
        self.save_src_html()
        # 另一张图清除之前锚点的比较结果
        self._dst_anchor_manager.clear()
        self._dst_color_manager.clear()
        self._dst_color_manager.set_anchors_color({self._dst_anchor_manager.get_first_level_anchor().get_anchor()})
        self.save_dst_html()
        # 清除比较标记
        self._is_first_level_anchor_compared = False
        self._is_second_level_anchor_compared.clear()

    def update_dst_executing_order(self):
        self._dst_executing_order: ExecutingOrder = ExecutingOrder(self._dst_executing_order_file_path)
        self._dst_anchor_manager: AnchorManager = AnchorManager(self._dst_executing_order)
        self._dst_color_manager: ColorManager = ColorManager()
        self._dst_color_manager.set_anchors_color({self._dst_anchor_manager.get_first_level_anchor().get_anchor()})
        self._dst_show_graph = _get_show_graph(
            self._dst_executing_order.get_graph(),
            self._dst_anchor_manager.get_topo_layers(),
            self._show_layer_num,
            self._show_all_layer
        )
        self.save_dst_html()
        # 另一张图清除之前锚点的比较结果
        self._src_anchor_manager.clear()
        self._src_color_manager.clear()
        self._src_color_manager.set_anchors_color({self._src_anchor_manager.get_first_level_anchor().get_anchor()})
        self.save_src_html()
        # 清除比较标记
        self._is_first_level_anchor_compared = False
        self._is_second_level_anchor_compared.clear()

    def update_src_first_level_anchor(self, anchor: Node):
        self._src_anchor_manager.update_first_level_anchor(anchor)
        self._src_color_manager.clear()
        self._src_color_manager.set_anchors_color({self._src_anchor_manager.get_first_level_anchor().get_anchor()})
        self._src_show_graph = _get_show_graph(
            self._src_executing_order.get_graph(),
            self._src_anchor_manager.get_topo_layers(),
            self._show_layer_num,
            self._show_all_layer
        )
        self.save_src_html()
        # 另一张图清除之前锚点的比较结果
        self._dst_anchor_manager.clear()
        self._dst_color_manager.clear()
        self._dst_color_manager.set_anchors_color({self._dst_anchor_manager.get_first_level_anchor().get_anchor()})
        self.save_dst_html()
        # 清除比较标记
        self._is_first_level_anchor_compared = False
        self._is_second_level_anchor_compared.clear()

    def update_dst_first_level_anchor(self, anchor: Node):
        self._dst_anchor_manager.update_first_level_anchor(anchor)
        self._dst_color_manager.clear()
        self._dst_color_manager.set_anchors_color({self._dst_anchor_manager.get_first_level_anchor().get_anchor()})
        self._dst_show_graph = _get_show_graph(
            self._dst_executing_order.get_graph(),
            self._dst_anchor_manager.get_topo_layers(),
            self._show_layer_num,
            self._show_all_layer
        )
        self.save_dst_html()
        # 另一张图清除之前锚点的比较结果
        self._src_anchor_manager.clear()
        self._src_color_manager.clear()
        self._src_color_manager.set_anchors_color({self._src_anchor_manager.get_first_level_anchor().get_anchor()})
        self.save_src_html()
        # 清除比较标记
        self._is_first_level_anchor_compared = False
        self._is_second_level_anchor_compared.clear()

    def fuse_src_nodes(self, fuse_nodes_id: list[int], peer_node: Node, fused_node_operator_type: str):
        fused_node = self._src_executing_order.fuse_nodes(fuse_nodes_id, peer_node, fused_node_operator_type)

        # 融合节点包含锚点时，需要更新锚点为融合后节点
        fuse_nodes = []
        for node_id in fuse_nodes_id:
            fuse_nodes.append(self._src_executing_order.get_node_manager().get_node_by_node_id(node_id))
        old_first_level_anchor = self._src_anchor_manager.get_first_level_anchor().get_anchor()
        new_first_level_anchor = old_first_level_anchor
        if old_first_level_anchor in fuse_nodes:
            new_first_level_anchor = fused_node
        
        self.update_src_first_level_anchor(new_first_level_anchor)

        # 清除比较标记
        self._is_first_level_anchor_compared = False
        self._is_second_level_anchor_compared.clear()

        return fused_node.get_node_id()

    def fuse_dst_nodes(self, fuse_nodes_id: list[int], peer_node: Node, fused_node_operator_type: str):
        fused_node = self._dst_executing_order.fuse_nodes(fuse_nodes_id, peer_node, fused_node_operator_type)

        # 融合节点包含锚点时，需要更新锚点为融合后节点
        fuse_nodes = []
        for node_id in fuse_nodes_id:
            fuse_nodes.append(self._dst_executing_order.get_node_manager().get_node_by_node_id(node_id))
        old_first_level_anchor = self._dst_anchor_manager.get_first_level_anchor().get_anchor()
        new_first_level_anchor = old_first_level_anchor
        if old_first_level_anchor in fuse_nodes:
            new_first_level_anchor = fused_node

        self.update_dst_first_level_anchor(new_first_level_anchor)

        # 清除比较标记
        self._is_first_level_anchor_compared = False
        self._is_second_level_anchor_compared.clear()

        return fused_node.get_node_id()
    
    def set_layer_num(self, layer_num: int):
        self._show_layer_num = layer_num
        self._src_show_graph = _get_show_graph(
            self._src_executing_order.get_graph(),
            self._src_anchor_manager.get_topo_layers(),
            self._show_layer_num,
            self._show_all_layer
        )
        self.save_src_html()
        self._dst_show_graph = _get_show_graph(
            self._dst_executing_order.get_graph(),
            self._dst_anchor_manager.get_topo_layers(),
            self._show_layer_num,
            self._show_all_layer
        )
        self.save_dst_html()

    def add_second_level_anchor(self, src_node: Node, dst_node: Node):
        self._src_anchor_manager.add_second_level_anchor(src_node)
        self._src_color_manager.set_anchors_color({src_node})
        self.save_src_html()
        self._dst_anchor_manager.add_second_level_anchor(dst_node)
        self._dst_color_manager.set_anchors_color({dst_node})
        self.save_dst_html()
        self._is_second_level_anchor_compared.append(False)

    def _compare_first_level_anchor(self):
        if not self._is_first_level_anchor_compared:
            gc = GraphComparatorAnchor(
                self._src_executing_order.get_graph(),
                self._dst_executing_order.get_graph(),
                self._src_anchor_manager.get_topo_layers(),
                self._dst_anchor_manager.get_topo_layers(),
            )
            src_match_nodes, dst_match_nodes, src_mismatch_nodes, dst_mismatch_nodes = gc.compare_graphs()
            self._src_anchor_manager.get_first_level_anchor().add_match_nodes(src_match_nodes)
            self._src_anchor_manager.get_first_level_anchor().add_mismatch_nodes(src_mismatch_nodes)
            self._dst_anchor_manager.get_first_level_anchor().add_match_nodes(dst_match_nodes)
            self._dst_anchor_manager.get_first_level_anchor().add_mismatch_nodes(dst_mismatch_nodes)
            self._src_color_manager.set_match_nodes_color(src_match_nodes)
            self._src_color_manager.set_mismatch_nodes_color(src_mismatch_nodes)
            self._dst_color_manager.set_match_nodes_color(dst_match_nodes)
            self._dst_color_manager.set_mismatch_nodes_color(dst_mismatch_nodes)
            self._is_first_level_anchor_compared = True

    def _compare_second_level_anchor(self):
        for anchor_index in range(len(self._is_second_level_anchor_compared)):
            if not self._is_second_level_anchor_compared[anchor_index]:
                src_anchor = self._src_anchor_manager.get_second_level_anchor(anchor_index).get_anchor()
                dst_anchor = self._dst_anchor_manager.get_second_level_anchor(anchor_index).get_anchor()
                src_topo_layers, _ = get_anchor_around_nodes_pos(
                    self._src_executing_order.get_graph(),
                    src_anchor,
                    NodePosition(0, 0, 0, 0),
                    False,
                    1,
                    -1
                )
                dst_topo_layers, _ = get_anchor_around_nodes_pos(
                    self._dst_executing_order.get_graph(),
                    dst_anchor,
                    NodePosition(0, 0, 0, 0),
                    False,
                    1,
                    -1
                )
                gc = GraphComparatorAnchor(
                    self._src_executing_order.get_graph(),
                    self._dst_executing_order.get_graph(),
                    src_topo_layers,
                    dst_topo_layers,
                )
                src_match_nodes, dst_match_nodes, src_mismatch_nodes, dst_mismatch_nodes = gc.compare_graphs()
                self._src_anchor_manager.get_second_level_anchor(anchor_index).add_match_nodes(src_match_nodes)
                self._src_anchor_manager.get_second_level_anchor(anchor_index).add_mismatch_nodes(src_mismatch_nodes)
                self._dst_anchor_manager.get_second_level_anchor(anchor_index).add_match_nodes(dst_match_nodes)
                self._dst_anchor_manager.get_second_level_anchor(anchor_index).add_mismatch_nodes(dst_mismatch_nodes)
                self._src_color_manager.set_match_nodes_color(src_match_nodes)
                self._src_color_manager.set_mismatch_nodes_color(src_mismatch_nodes)
                self._dst_color_manager.set_match_nodes_color(dst_match_nodes)
                self._dst_color_manager.set_mismatch_nodes_color(dst_mismatch_nodes)
                self._is_second_level_anchor_compared[anchor_index] = True

    def compare_graphs(self):
        self._compare_first_level_anchor()
        self._compare_second_level_anchor()

        self._src_color_manager.set_anchors_color(self._src_anchor_manager.get_all_anchors())
        self._dst_color_manager.set_anchors_color(self._dst_anchor_manager.get_all_anchors())

        self.save_src_html()
        self.save_dst_html()
