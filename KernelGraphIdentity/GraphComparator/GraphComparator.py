import networkx as nx


class GraphComparator:
    """
    比较两个有向图的差异。
    """
    def __init__(self, src_graph: nx.DiGraph, dst_graph: nx.DiGraph):
        """
        :param src_graph: 第一个有向图 (nx.DiGraph)
        :param dst_graph: 第二个有向图 (nx.DiGraph)
        """
        self._src_graph = src_graph
        self._dst_graph = dst_graph

    def compare_graphs(self):
        """
        比较两个有向图的差异。

        :return: 返回差异结果
        """
        pass
