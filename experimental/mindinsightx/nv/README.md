# Mindspore NetworkValidator

* [一、介绍](#jump1)
* [二、概念](#jump2)
	* [1. 计算图](#jump3)
	* [2. 节点映射](#jump4)
* [三、安装](#jump5)
* [四、使用方法](#jump6)
	* [1. 导出计算图](#jump7)
	* [2. 对比计算图](#jump8)
	* [3. dump文件比较说明](#jump9)
* [五、进阶使用](#jump10)
	* [1. 多种API使用策略](#jump11)
	* [2. mappings映射表说明](#jump12)


## <span id="jump1">一、介绍</span>

Mindspore NetworkValidator是一个同框架下(MindSpore)不同后端/版本深度学习模型中节点精度的自动化对比工具，为开发人员在模型网络、框架或硬件迁移时提供前后对比及除错功能。

## <span id="jump2">二、概念</span>
### <span id="jump3">1. 计算图</span>

在深度学习中，我们可用一个计算图去代表一个模型网络。计算图是一种有向图，包含了节点及有向边。每个节点代表一个具有特定输出张量维度大小(例如3x56x56)的算子(例如conv2d，relu等)，算子类型和输出张量维度为算子属性，而有向边侧代表了张量数据的流向。
### <span id="jump4">2. 节点映射</span>

依靠算子属性及边结构为两个计算图的节点找出映射关系是找出两图之间结构差异的最重要一步。这一找出映射关系的过程可能非常耗时，所以本工具的首要目标就是为这过程提供一个自动化的解决方案。

## <span id="jump5">三、安装</span>

若想做好映射关系对比，安装好本工具至关重要，请参照以下指引进行安装。
1. 首先，根据requirements配置好对应环境；
```python
pip install requirements.txt
```
2. 其次，若你还没有安装Mindspore，根据你想对比的Mindspore模型，去[官网](https://www.mindspore.cn/install)搭建对应的版本，例如安装Mindspore 2.1.1版本等；
3. 之后，在使用前，将当前路径调节至./mindinsightx的上一级路径中，随后可调用./mindinsightx/nv/compare.py文件进行使用；
4. 最后，该框架仅支持Mindspore框架下的比较，具体如以下表格所示。

| 标号 | 框架 |
| :----:| :----: |
| me-XXX | MindSpore Ascend |
| me-XXX-gpu | MindSpore GPU |

## <span id="jump6">四、使用方法</span>
### <span id="jump7">1. 导出计算图</span>

该框架已适配不同类型的图比较，如vm图和ge图之间的比较，其中vm图以pb文件格式加载，ge图以pbtxt文件格式加载，逻辑体现在me_graph.py文件的load函数中，感兴趣的用户可自行查看。
#### VM图

在VM图中，用户可以如下方例子般在训练模型时导出计算图：
```python
from mindspore import context

context.set_context(save_graphs=True, save_graphs_path='./graphs')
# then train a network with 1 epoch ...
```
训练完成后，会有数个.pb文件产生在 "./graphs/" 文件夹中：
```python3
ms_output_after_opt_0.pb
ms_output_before_opt_0.pb
ms_output_optimize.pb
ms_output_vm_build_0.pb
```
其中，"ms_output_before_opt_0.pb" 是还没有进行任何硬件优化的计算图，保留了最多脚本相关的信息，可用于比较网络结构；“ms_output_after_opt_0.pb”是经过硬件优化的图，可用于比较dump转储。用户可以用以下方法加载：
```python
from mindinsight.nv.graph.graph import Graph

graph0 = Graph.create('me', '2.0.0')
graph0.load('./graphs/ms_output_before_opt_0.pb')
```

#### GE图

在GE图中，用户可提取文件名类似ge_onnx_00000000_graph_1_Build.pbtxt或ge_proto_00000000_graph_1_Build.txt的.pbtxt文件，内容如以下格式：
```python3
ir_version: 6
producer_name: "GE"
graph {
  node {
    input: ""
    output: "args0:0"
    name: "args0"
    op_type: "ge:Data"
    attribute {
      name: "OUTPUT_IS_VAR"
      ints: 0
      type: INTS
    }
    attribute {
      name: "OwnerGraphIsUnknown"
      type: INT
    }
```
其中，选取onnx类型的文件使用，“ge_onnx_00000000_graph_1_Build.pbtxt”保留了最多脚本相关的信息，可用于比较网络结构。用户可以用以下方法加载：
```python
from mindinsightx.nv.graph.graph import Graph

graph1 = Graph.create('me', '2.1.1')
graph1.load('./ge_onnx_00000000_graph_1_Build.pbtxt')
```

### <span id="jump8">2. 对比计算图</span>

Mindspore NetworkValidator暂时只提供Python API界面，以下是一个对比MindSpore v2.0.0 VM计算图和MindSpore v2.1.1 GE计算图的基本使用例子：
```python
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
comparator.save_as_xlsx('report.xlsx')
```
其中，`Graph.create(graph_type, framework_version)` 创建指定的深度学习框架计算图对象。同时，在使用时将路径调节至./mindinsightx的上一级路径中。
`comparator.compare()` 是一个优化过程，当内部目标函数数值被最大化后便会自动停止。优化过程会产生以下类似的列印信息：

![](https://gitee.com/bochen2023/toolkits/raw/master/toolkits/experimental/mindinsightx/nv/images/comparator_prints.png)

用户可以自行设定最大的优化步数 (默认为无限)：
```python
comparator.compare(max_steps=10)
```
打开最终输出保存的对比报告 "report.txt"，第一个表是概榄 "summary"。它会列出计算图的种类，各种算子的总算及被成功映射的算目。

![](https://gitee.com/bochen2023/toolkits/raw/master/toolkits/experimental/mindinsightx/nv/images/report_summary.png)

第二个表是"graph0 mapped"，会列出所有graph0(本例中为MindSpore的VM计算图)的算子及其在graph1(本例中为MindSpore的GE计算图)中的映射算子。最后一列"Similarity" 是两个算子之间的拓朴相似性，0.0代表完全不相似，1.0代表拓朴结构完全相同。如果没找到映射关系，映射算子相关的列便会被留空。

![](https://gitee.com/bochen2023/toolkits/raw/master/toolkits/experimental/mindinsightx/nv/images/graph0_mapped.png)

在本例子中graph0的算子名称非常长，是因为MindSpore算子的显示名称为"scope: name"格式，scope能告诉用户很多脚本及网络结构的信息。用户可用以下方法指示只显示算子的原始名称：
```python
comparator.save_as_xlsx('report.xlsx', show_scoped_name=False)
```

第三个表是 "graph0 top-5"，会列出所有graph0的算子及跟其在拓朴上最相似的五个graph1算子，最后一列Mapped会列出成功匹配算子的信息。

![](https://gitee.com/bochen2023/toolkits/raw/master/toolkits/experimental/mindinsightx/nv/images/graph0_topk.png)

用户可以用以下方法设定相似算子的显示数目：
```python
comparator.save_as_xlsx('report.xlsx', top_k=3)
```
之后两个表 "graph1 mapped" 及 "graph1 top-5" 跟 "graph0 mapped" 及 "graph0 top-5" 相同，只是换成graph1作为本位。

### <span id="jump9">3. dump文件比较说明</span>
  
为了比较运算符的精度，需要使用dump文件。dump文件的内容受数据集、网络结构和运算符实现的影响。因此，用户必须遵循一定的规则：
1. 数据集、样本顺序和批量大小必须相同。
2. 数据集内部不能有随机性，如随机裁剪、数据增强。
3. 网络和训练脚本必须相同，除非意图比较具有不同结构的网络。
4. 网络内部不能有随机性，如dropout层。
违反上述规则可能导致从转储比较中得出不正确的结论。

若想要使用dump文件进行比较，用户必须首先创建一个名为 data_dump.json 的 JSON 文件：
```python
{
    "common_dump_settings": {
        "dump_mode": 0,
        "path": "./mindinsightx/nv/dump_path",
        "net_name": "alexnet",
        "iteration": "0",
        "input_output": 0,
        "kernels": [],
        "support_device": [0,1,2,3,4,5,6,7]
    },
    "e2e_dump_settings": {
        "enable": true,
        "trans_flag": true
    }
}
```
dump_mode 和 input_output 必须设置为 0，以便转储所有运算符的所有输入和输出。通常比较第一到第二次迭代，因此，将iteration设置为 0 或 1。
  
然后指定环境变量：
```python
export MINDSPORE_DUMP_CONFIG=/data_dump.json
```
运行训练脚本后，将输出dump文件格式如下所示：
```python3
BiasAdd.Default_BiasAdd-op5Default_ReLU-op7.15.21.1697720863549166.input.0.NCHW.npy
BiasAddGrad.Gradients_Default_gradBiasAdd-expand_BiasAddGrad-op63.115.21.1697720869436202.input.0.NCHW.npy
Conv2D.Default_Conv2D-op3.14.21.1697720863520127.input.0.NCHW.npy
Conv2DBackpropFilterD.Default_Conv2DBackpropFilter-op100.111.21.1697720869347459.input.0.NCHW.npy
MatMulV2.Default_MatMul-op34Default_ReLU-op38.42.21.1697720864230650.input.0.FRACTAL_NZ.npy
```
随后，若要使用dump文件进行对比，需在compare.py文件中添加dump文件的路径，如下所示：
```python
# 原对比逻辑
# comparator = GraphComparator(graph0, graph1)
# 添加dump文件后对比逻辑
comparator = GraphComparator(graph0, graph1, "graph0_dump_dir", "graph1_dump_dir")
```

需要注意的是，在添加使用dump文件映射过程中， graph0_dump_dir 和 graph1_dump_dir 是包含单次迭代的dump文件的直接dump目录，不会搜索子目录。该框架支持同步和异步dump，但只比较输出dump。

若用户添加使用了dump文件的对比策略，在输出report.xlsx表时会多增加两个表格graph0 dump和graph1 dump，它们分别对应graph0和graph1中的dump文件映射情况，每一个节点所对应的dump文件会被呈现在”dump“项中，若没有找到该节点的dump文件，则会出现”missing“字样，如下图所示：

![](https://gitee.com/bochen2023/toolkits/raw/master/toolkits/experimental/mindinsightx/nv/images/graph1_dump.png)

## <span id="jump10">五、进阶使用</span>
### <span id="jump11">1. 多种API使用策略</span>

Mindspore NetworkValidator提供Python API接口。以下是比较MindSpore v2.0.0网络VM类型的图和MindSpore v2.1.1网络GE类型的图的最基本示例。
```python
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
```

除此之外，如果用户希望直接从源代码中使用命令行工具而不安装 MindInsight，可以通过调用cli.py的方式进行运行，示例如下：
```python
python mindinsightx/nv/cli.py --graph-file /ms_output_before_opt_0.pb
                              --graph-type me-2.0.0-gpu 
                              --cmp-graph-file /your_ge_graph_path.pbtxt
                              --cmp-graph-type me-2.1.0-gpu 
                              --dump-dir /graph0_dump_dir
                              --cmp-dump-dir /graph1_dump_dir
                              --report /your_report_path/xxx.xlsx
```

### <span id="jump12">2. mappings映射表说明</span>

在加载GE图生成的节点文件时发现，存在少数节点类型type存在后缀的情况，如'Conv2DBackpropFilterD'等。这与正常节点类型不同，将导致本该成功匹配的节点匹配失败，故在./mindinsightx/nv/mapper/mappings.py文件中设置mappings映射表，若在对比过程中发现相同类似错误情况时，可直接在表中以 **’错误节点类型’: ’正确节点类型’** 的形式将问题节点添加，以解决此类问题。格式如下：
```python
mappings = {
    'Conv2DBackpropFilterD': 'Conv2DBackpropFilter',
    'Conv2DBackpropInputD': 'Conv2DBackpropInput'
}
```


