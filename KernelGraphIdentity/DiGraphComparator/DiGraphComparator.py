import networkx as nx


class DiGraphComparator:
    """
    有向图差异比较器。
    """
    def __init__(self, src_graph: nx.DiGraph, dst_graph: nx.DiGraph):
        self._src_graph = src_graph
        self._dst_graph = dst_graph

    def compare_graphs(self):
        """
        比较两个有向图的差异。
        """
        pass
