import sys
import os

projectPath = os.path.abspath(os.path.join(os.getcwd()))
print(projectPath)
sys.path.append(projectPath)

from mindinsightx.nv.graph.graph import Graph
from mindinsightx.nv.comparator.graph_comparator import GraphComparator

graph0 = Graph.create('me', '2.0.0-gpu')
graph0.load('your_vm_graph_path/xxx.pb')
graph1 = Graph.create('me', '2.1.0-gpu')
graph1.load('your_ge_graph_path/xxx.pbtxt')

comparator = GraphComparator(graph0, graph1)
comparator.compare()
comparator.save_as_xlsx('your_report_path/xxx.xlsx')
