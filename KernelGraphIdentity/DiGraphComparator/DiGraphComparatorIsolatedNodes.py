from DiGraphComparator.DiGraphComparator import DiGraphComparator
from ExecutingOrder.NodeManager import Node


class DiGraphComparatorIsolatedNodes(DiGraphComparator):
    def __init__(self, src_graph, dst_graph):
        super().__init__(src_graph, dst_graph)

    @staticmethod
    def _record_key_nodes(record: dict[str, set[Node]], key: str, node: Node):
        """
        记录key值对应的节点。
        """
        record.setdefault(key, set())
        record[key].add(node)

    @staticmethod
    def _record_key_times(record: dict[str, int], key: str):
        """
        记录key值出现次数。
        """
        record.setdefault(key, 0)
        record[key] += 1

    @staticmethod
    def _get_only_in_src_nodes_info(info_lines: list[str], key: str, src_key_nodes: dict[str, set[Node]]):
        info_lines.append("如下算子特征存在差异：")
        node = next(iter(src_key_nodes[key]))
        info_lines.append(" " * 2 + f"算子类型：{node.get_operator_type()}")

        input_memory_attributes = list(str(memory_attribute)
                                       for memory_attribute in node.get_input_memory_attributes())
        input_memory_attributes_str = ", ".join(input_memory_attributes)
        info_lines.append(" " * 2 + f"输入数量：{len(input_memory_attributes)}")
        info_lines.append(" " * 2 + f"输入属性：{input_memory_attributes_str}")

        output_memory_attributes = list(str(memory_attribute)
                                        for memory_attribute in node.get_output_memory_attributes())
        output_memory_attributes_str = ", ".join(output_memory_attributes)
        info_lines.append(" " * 2 + f"输出数量：{len(output_memory_attributes)}")
        info_lines.append(" " * 2 + f"输出属性：{output_memory_attributes_str}")

        info_lines.append(" " * 2 + f"左图该特征算子数量为：{len(src_key_nodes[key])}")
        info_lines.append(" " * 2 + "右图该特征算子数量为：0")
        info_lines.append(" " * 2 + "左图该特征算子有：")
        for node in src_key_nodes[key]:
            info_lines.append(" " * 4 + f"算子类型 {node.get_operator_type()} 行号 {node.get_line_num()}")

        
        info_lines.append("")

    @ staticmethod
    def _get_only_in_dst_nodes_info(info_lines: list[str], key: str, dst_key_nodes: dict[str, set[Node]]):
        info_lines.append("如下算子特征存在差异：")
        node = next(iter(dst_key_nodes[key]))
        info_lines.append(" " * 2 + f"算子类型：{node.get_operator_type()}")

        input_memory_attributes = list(str(memory_attribute)
                                       for memory_attribute in node.get_input_memory_attributes())
        input_memory_attributes_str = ", ".join(input_memory_attributes)
        info_lines.append(" " * 2 + f"输入数量：{len(input_memory_attributes)}")
        info_lines.append(" " * 2 + f"输入属性：{input_memory_attributes_str}")

        output_memory_attributes = list(str(memory_attribute)
                                        for memory_attribute in node.get_output_memory_attributes())
        output_memory_attributes_str = ", ".join(output_memory_attributes)
        info_lines.append(" " * 2 + f"输出数量：{len(output_memory_attributes)}")
        info_lines.append(" " * 2 + f"输出属性：{output_memory_attributes_str}")

        info_lines.append(" " * 2 + "左图该特征算子数量为：0")
        info_lines.append(" " * 2 + f"右图该特征算子数量为：{len(dst_key_nodes[key])}")
        info_lines.append(" " * 2 + "右图该特征算子有：")
        for node in dst_key_nodes[key]:
            info_lines.append(" " * 4 + f"算子类型 {node.get_operator_type()} 行号 {node.get_line_num()}")

    @staticmethod
    def _get_key_times_diff_nodes_info( 
            info_lines: list[str],
            key: str,
            src_key_nodes: dict[str, set[Node]],
            dst_key_nodes: dict[str, set[Node]]
    ):
        info_lines.append("如下算子特征存在差异：")
        node = next(iter(src_key_nodes[key]))
        info_lines.append(" " * 2 + f"算子类型：{node.get_operator_type()}")

        input_memory_attributes = list(str(memory_attribute)
                                       for memory_attribute in node.get_input_memory_attributes())
        input_memory_attributes_str = ", ".join(input_memory_attributes)
        info_lines.append(" " * 2 + f"输入数量：{len(input_memory_attributes)}")
        info_lines.append(" " * 2 + f"输入属性：{input_memory_attributes_str}")

        output_memory_attributes = list(str(memory_attribute)
                                        for memory_attribute in node.get_output_memory_attributes())
        output_memory_attributes_str = ", ".join(output_memory_attributes)
        info_lines.append(" " * 2 + f"输出数量：{len(output_memory_attributes)}")
        info_lines.append(" " * 2 + f"输出属性：{output_memory_attributes_str}")

        info_lines.append(" " * 2 + f"左图该特征算子数量为：{len(src_key_nodes[key])}")
        info_lines.append(" " * 2 + f"右图该特征算子数量为：{len(dst_key_nodes[key])}")
        info_lines.append(" " * 2 + "左图该特征算子有：")
        for node in src_key_nodes[key]:
            info_lines.append(" " * 4 + f"算子类型 {node.get_operator_type()} 行号 {node.get_line_num()}")

        
        info_lines.append(" " * 2 + "右图该特征算子有：")
        for node in dst_key_nodes[key]:
            info_lines.append(" " * 4 + f"算子类型 {node.get_operator_type()} 行号 {node.get_line_num()}")

    def compare_graphs(self, src_isolated_nodes: list[Node], dst_isolated_nodes:list[Node]) -> list[str]:
        info_lines: list[str] = []
        src_key_times: dict[str, int] = {}  # 记录key值出现的次数
        src_key_nodes: dict[str, set[Node]] = {}  # 记录key值对应节点
        for node in src_isolated_nodes:
            key = node.get_node_feature()
            DiGraphComparatorIsolatedNodes._record_key_times(src_key_times, key)
            DiGraphComparatorIsolatedNodes._record_key_nodes(src_key_nodes, key, node)

        dst_key_times: dict[str, int] = {}  # 记录key值出现的次数
        dst_key_nodes: dict[str, set[Node]] = {}  # 记录key值对应节点
        for node in dst_isolated_nodes:
            key = node.get_node_feature()
            DiGraphComparatorIsolatedNodes._record_key_times(dst_key_times, key)
            DiGraphComparatorIsolatedNodes._record_key_nodes(dst_key_nodes, key, node)

        for key in set(src_key_times.keys()) | set(dst_key_times.keys()):
            if dst_key_times.get(key) is None:  # key值只在src图中存在
                DiGraphComparatorIsolatedNodes._get_only_in_src_nodes_info(info_lines, key, src_key_nodes)
                info_lines.append("")
                continue
            if src_key_times.get(key) is None:  # key值只在dst图中存在
                DiGraphComparatorIsolatedNodes._get_only_in_dst_nodes_info(info_lines, key, dst_key_nodes)
                info_lines.append("")
                continue
            if src_key_times[key] != dst_key_times[key]:  # key值在两图中出现次数不一致
                DiGraphComparatorIsolatedNodes._get_key_times_diff_nodes_info(
                    info_lines,
                    key,
                    src_key_nodes,
                    dst_key_nodes
                )
                info_lines.append("")
                continue

        return info_lines
