from ExecutingOrder.MemoryAttribute import MemoryAttribute
from ExecutingOrder.NodeManager import Node


class Edge:
    def __init__(self):
        self._memory_id_attributes: dict[str, list[MemoryAttribute]] = {}  # 记录边上的内存id和内存id的属性列表

    def update_memory_id_attributes(self, memory_id: str, attributes: list[MemoryAttribute]):
        self._memory_id_attributes.setdefault(memory_id, [])
        self._memory_id_attributes[memory_id].extend(attributes)

    def get_memory_id_attributes(self) -> dict[str, list[MemoryAttribute]]:
        return self._memory_id_attributes

    def __repr__(self):
        return "\n".join(
            (
                memory_id + "\n" +
                "\n".join([str(memory_attribute) for memory_attribute in memory_attributes])
            )
            for memory_id, memory_attributes in self._memory_id_attributes.items()
        )


class EdgeManager:
    def __init__(self):
        self._edge_key_to_edge: dict[str, Edge] = {}

    @staticmethod
    def _get_edge_key(src_node: Node, dst_node: Node) -> str:
        return f"{src_node.get_node_id()}->{dst_node.get_node_id()}"

    def get_edge(self, src_node: Node, dst_node: Node) -> Edge:
        key = EdgeManager._get_edge_key(src_node, dst_node)
        return self._edge_key_to_edge.get(key)

    def add_edge(self, src_node: Node, dst_node: Node):
        key = EdgeManager._get_edge_key(src_node, dst_node)
        self._edge_key_to_edge.setdefault(key, Edge())

    def update_edge(self, src_node: Node, dst_node: Node, memory_id: str, attributes: list[MemoryAttribute]):
        key = EdgeManager._get_edge_key(src_node, dst_node)
        self._edge_key_to_edge.setdefault(key, Edge())

        edge = self._edge_key_to_edge[key]
        edge.update_memory_id_attributes(memory_id, attributes)

    def update_edge_by_other_edge(self, src_node: Node, dst_node: Node, edge: Edge):
        for memory_id, attributes in edge.get_memory_id_attributes().items():
            self.update_edge(src_node, dst_node, memory_id, attributes)

    def add_edge_for_fuse(
            self,
            fused_node: Node,
            successors_old_nodes: dict[Node, list[Node]],
            predecessors_old_nodes: dict[Node, list[Node]]
    ):
        for successor, nodes in successors_old_nodes.items():
            for node in nodes:
                edge = self.get_edge(node, successor)
                self.update_edge_by_other_edge(fused_node, successor, edge)
        for predecessor, nodes in predecessors_old_nodes.items():
            for node in nodes:
                edge = self.get_edge(predecessor, node)
                self.update_edge_by_other_edge(predecessor, fused_node, edge)
