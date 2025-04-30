from ExecutingOrder.NodeManager import Node


DEFAULT_NODE_BACKGROUND_COLOR = "#97c2fc"  # 浅蓝色
DEFAULT_NODE_BORDER_COLOR = "#97c2fc"  # 浅蓝色
MATCH_NODE_BACKGROUND_COLOR = "green"
MATCH_NODE_BORDER_COLOR = "green"
MISMATCH_NODE_BACKGROUND_COLOR = "red"
MISMATCH_NODE_BORDER_COLOR = "red"
ANCHOR_NODE_BACKGROUND_COLOR = "orange"
ANCHOR_NODE_BORDER_COLOR = "orange"
SELECTED_NODE_BACKGROUND_COLOR = "yellow"
SELECTED_NODE_BORDER_COLOR = "black"


class NodeColor:
    def __init__(self, background_color: str, border_color: str):
        self.background_color = background_color
        self.border_color = border_color


class ColorManager:
    def __init__(self):
        self._node_color: dict[Node, NodeColor] = {}

    def set_match_nodes_color(self, match_nodes: set[Node]):
        for node in match_nodes:
            self._node_color[node] = NodeColor(MATCH_NODE_BACKGROUND_COLOR, MATCH_NODE_BORDER_COLOR)

    def set_mismatch_nodes_color(self, mismatch_nodes: set[Node]):
        for node in mismatch_nodes:
            self._node_color[node] = NodeColor(MISMATCH_NODE_BACKGROUND_COLOR, MISMATCH_NODE_BORDER_COLOR)

    def set_anchors_color(self, anchors: set[Node]):
        for anchor in anchors:
            self._node_color[anchor] = NodeColor(ANCHOR_NODE_BACKGROUND_COLOR, ANCHOR_NODE_BORDER_COLOR)

    def get_node_color(self, node: Node) -> NodeColor:
        if self._node_color.get(node) is not None:
            return self._node_color[node]
        return NodeColor(DEFAULT_NODE_BACKGROUND_COLOR, DEFAULT_NODE_BORDER_COLOR)

    def get_edge_color(self, src_node: Node, dst_node: Node) -> str:
        # 边颜色与指向的节点颜色保持一致
        return self.get_node_color(dst_node).background_color

    def clear(self):
        self._node_color.clear()
