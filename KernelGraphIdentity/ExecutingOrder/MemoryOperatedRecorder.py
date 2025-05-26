from ExecutingOrder.MemoryAttribute import MemoryAttribute
from ExecutingOrder.NodeManager import Node


class Operation:
    def __init__(
            self,
            node: Node,
            memory_attribute: MemoryAttribute,
            input_or_output: str
    ):
        self._node = node  # 操作的node
        self._memory_attribute = memory_attribute  # 操作的内存id属性
        self._input_or_output = input_or_output  # 操作类型是输入还是输出


class MemoryOperatedRecorder:
    """
    内存被操作记录。
    """
    def __init__(self):
        self._record: dict[str, list[Operation]] = {}

    def record_operation(
            self,
            memory_id: str,
            node: Node,
            memory_attribute: MemoryAttribute,
            input_or_output: str
    ):
        self._record.setdefault(memory_id, [])
        operation = Operation(node, memory_attribute, input_or_output)
        self._record[memory_id].append(operation)
