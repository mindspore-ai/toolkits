from ExecutingOrder.MemoryAttribute import MemoryAttribute


VIRTUAL_NODE_OPERATOR_TYPE = "VirtualNode"


class Node:
    def __init__(self):
        self._node_id: int = -1
        self._operator_type: str = ""
        self._input_memory_ids: list[str] = []
        self._output_memory_ids: list[str] = []
        self._input_memory_attributes: list[MemoryAttribute] = []
        self._output_memory_attributes: list[MemoryAttribute] = []
        self._task_info: str = ""
        self._attrs: str = ""
        self._line_num: int = -1
        self._node_feature: str = ""

    def set_node_id(self, node_id: int):
        self._node_id = node_id

    def get_node_id(self) -> int:
        return self._node_id

    def set_operator_type(self, operator_type: str):
        self._operator_type = operator_type

    def get_operator_type(self) -> str:
        return self._operator_type

    def set_input_memory_ids(self, input_memory_ids: list[str]):
        self._input_memory_ids = input_memory_ids

    def get_input_memory_ids(self) -> list[str]:
        return self._input_memory_ids

    def set_output_memory_ids(self, output_memory_ids: list[str]):
        self._output_memory_ids = output_memory_ids

    def get_output_memory_ids(self) -> list[str]:
        return self._output_memory_ids

    def set_input_memory_attributes(self, input_memory_attributes: list[MemoryAttribute]):
        self._input_memory_attributes = input_memory_attributes

    def get_input_memory_attributes(self) -> list[MemoryAttribute]:
        return self._input_memory_attributes

    def set_output_memory_attributes(self, output_memory_attributes: list[MemoryAttribute]):
        self._output_memory_attributes = output_memory_attributes

    def get_output_memory_attributes(self) -> list[MemoryAttribute]:
        return self._output_memory_attributes

    def set_task_info(self, task_info: str):
        self._task_info = task_info

    def get_task_info(self) -> str:
        return self._task_info

    def set_attrs(self, attrs: str):
        self._attrs = attrs

    def get_attrs(self) -> str:
        return self._attrs

    def set_line_num(self, line_num: int):
        self._line_num = line_num

    def get_line_num(self) -> int:
        return self._line_num

    def set_node_feature(self, node_feature: str):
        self._node_feature = node_feature

    def get_node_feature(self) -> str:
        return self._node_feature

    # Add 4 1
    def __repr__(self):
        return f"{self._operator_type} {self._line_num} {self._node_id}"

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.__repr__() == other.__repr__()
        return False


class NodeManager:
    def __init__(self):
        self._max_node_id = 0  # 节点id最小值为1，这里记录最大值
        self._node_id_to_node: dict[int, Node] = {}

    def add_node(self, node: Node):
        self._max_node_id += 1
        node.set_node_id(self._max_node_id)
        self._node_id_to_node[node.get_node_id()] = node

    def get_node_by_node_id(self, node_id: int) -> Node:
        return self._node_id_to_node[node_id]

    def get_node_by_operator_type(self, operator_type: str) -> Node:
        for node in self._node_id_to_node.values():
            if node.get_operator_type() == operator_type:
                return node
            
    def get_node_by_line_num(self, line_num: int) -> Node:
        for node in self._node_id_to_node.values():
            if node.get_line_num() == line_num:
                return node

    def get_first_node(self) -> Node:
        return self.get_node_by_node_id(1)
    
    def get_virtual_node(self) -> Node:
        return self.get_node_by_operator_type(VIRTUAL_NODE_OPERATOR_TYPE)

    def get_all_nodes(self) -> list[Node]:
        all_nodes = []
        for node_id in range(1, self._max_node_id + 1):
            all_nodes.append(self.get_node_by_node_id(node_id))
        return all_nodes
