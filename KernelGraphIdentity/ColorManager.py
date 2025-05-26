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

SELECTED_EDGE_COLOR = "black"
DEFAULT_EDGE_COLOR = "#97c2fc"  # 浅蓝色


class NodeColor:
    def __init__(self, background_color: str, border_color: str):
        self.background_color = background_color
        self.border_color = border_color


class ColorManager:
    def __init__(self):
        pass

    @staticmethod
    def get_matched_node_color() -> NodeColor:
        return NodeColor(MATCH_NODE_BACKGROUND_COLOR, MATCH_NODE_BORDER_COLOR)
    
    @staticmethod
    def get_mismatched_node_color() -> NodeColor:
        return NodeColor(MISMATCH_NODE_BACKGROUND_COLOR, MISMATCH_NODE_BORDER_COLOR)
    
    @staticmethod
    def get_anchor_node_color() -> NodeColor:
        return NodeColor(ANCHOR_NODE_BACKGROUND_COLOR, ANCHOR_NODE_BORDER_COLOR)

    @staticmethod
    def get_default_node_color() -> NodeColor:
        return NodeColor(DEFAULT_NODE_BACKGROUND_COLOR, DEFAULT_NODE_BORDER_COLOR)

    @staticmethod
    def get_edge_color(src_node: Node, dst_node: Node) -> str:
        return DEFAULT_EDGE_COLOR
