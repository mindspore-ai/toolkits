from ExecutingOrder.MemoryAttribute import MemoryAttribute

SOURCE_VIRTUAL_NODE_OPERATOR_TYPE = "SourceVirtualNode"
SINK_VIRTUAL_NODE_OPERATOR_TYPE = "SinkVirtualNode"
INVALID_NODE_ID = -1  # 无效的节点id


class Node:
    def __init__(self):
        self._node_id: int = -1
        self._operator_type: str = ""
        self._full_scope: list[str] = []
        self._stack: list[str] = []
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

    def set_full_scope(self, full_scope: list[str]):
        self._full_scope = full_scope

    def get_full_scope(self) -> list[str]:
        return self._full_scope

    def set_stack(self, stack: list[str]):
        self._stack = stack

    def get_stack(self) -> list[str]:
        return self._stack

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

    def get_node_base_info(self, padding: str) -> list[str]:
        info_lines: list[str] = []

        info_lines.append(padding + "节点基础信息：")

        info_lines.append(padding + " " * 2 + f"算子类型：{self.get_operator_type()}")

        input_memory_attributes = list(str(memory_attribute)
                                       for memory_attribute in self.get_input_memory_attributes())
        input_memory_attributes_str = ", ".join(input_memory_attributes)
        info_lines.append(padding + " " * 2 + f"输入数量：{len(input_memory_attributes)}")
        info_lines.append(padding + " " * 2 + f"输入属性：{input_memory_attributes_str}")

        output_memory_attributes = list(str(memory_attribute)
                                        for memory_attribute in self.get_output_memory_attributes())
        output_memory_attributes_str = ", ".join(output_memory_attributes)
        info_lines.append(padding + " " * 2 + f"输出数量：{len(output_memory_attributes)}")
        info_lines.append(padding + " " * 2 + f"输出属性：{output_memory_attributes_str}")

        return info_lines

    def get_full_scope_info(self, padding: str) -> list[str]:
        info_lines: list[str] = []
        info_lines.append(padding + "节点 full scope：")
        for scope in self._full_scope:
            info_lines.append(padding + " " * 2 + scope)
        return info_lines

    def get_stack_info(self, padding: str) -> list[str]:
        info_lines: list[str] = []
        info_lines.append(padding + "节点 stack：")
        for stack in self._stack:
            info_lines.append(padding + " " * 2 + stack)
        return info_lines

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
        self._node_num = 0
        self._node_id_to_node: dict[int, Node] = {}
        self._first_node_id = INVALID_NODE_ID

        self._source_virtual_node = Node()
        self._source_virtual_node.set_operator_type(SOURCE_VIRTUAL_NODE_OPERATOR_TYPE)
        self.add_node(self._source_virtual_node)

        self._sink_virtual_node = Node()
        self._sink_virtual_node.set_operator_type(SINK_VIRTUAL_NODE_OPERATOR_TYPE)
        self.add_node(self._sink_virtual_node)

        self._isolated_nodes: list[Node] = []

    def add_node(self, node: Node):
        node_id = self._node_num
        self._node_num += 1
        node.set_node_id(node_id)
        self._node_id_to_node[node_id] = node

    def get_node_by_node_id(self, node_id: int) -> Node:
        return self._node_id_to_node[node_id]

    def get_node_by_line_num(self, line_num: int) -> Node | None:
        for node in self._node_id_to_node.values():
            if node.get_line_num() == line_num:
                return node
        return None

    def get_all_nodes(self) -> list[Node]:
        return [self.get_node_by_node_id(node_id) for node_id in range(self._node_num)]

    def get_source_virtual_node(self) -> Node:
        return self._source_virtual_node

    def get_sink_virtual_node(self) -> Node:
        return self._sink_virtual_node

    def set_first_node(self, node: Node):
        self._first_node_id = node.get_node_id()

    def get_first_node(self) -> Node:
        return self._node_id_to_node[self._first_node_id]

    def record_isolated_node(self, node: Node):
        self._isolated_nodes.append(node)

    def get_isolated_nodes(self) -> list[Node]:
        return self._isolated_nodes
