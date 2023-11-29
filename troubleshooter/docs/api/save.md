## troubleshooter.save
>
> troubleshooter.save(file:str, data:Union(Tensor, list[Tensor], tuple[Tensor], dict[str, Tensor]), suffix=None，use_print=False)

`MindSpore`和`PyTorch`的统一数据保存接口，会将`Tensor`数据保存为numpy的npy格式文件。

### 参数

- file: 文件名路径。为避免同名文件覆盖，保存的文件名称会自动添加前缀，前缀从0开始，按照执行顺序递增。
- data: 数据，支持保存`Tensor`（包括`mindspore.Tensor`和`pytorch.tensor`），以及`Tensor`构成的`list/tuple/dict`。当为`list/tuple`类型时，会按照顺序添加编号；当为`dict`类型时，文件名中会添加`key`。
- suffix: 文件名后缀，默认值为`None`。
- use_print：是否使用Print算子落盘保存，仅在MindSpore框架Ascend硬件平台VM流程下有效。

### 文件名称格式

存储的文件名称为 **[id]\_name\.[idx/key]\_[suffix].npy**。
- id为按照执行顺序递增的前缀；
- name为file中指定的文件名；
- 当数据为嵌套类型时，会在名称后添加编号，使用点分隔；
- suffix为指定的后缀。

> **Warning:**
>
> - 在MindSpore 2.0版本中，save函数暂时不支持图模式。
>
> - 在2.1版本中，save函数支持MindSpore图模式，但实现依赖于[JIT Fallback](https://mindspore.cn/docs/zh-CN/master/design/dynamic_graph_and_static_graph.html#jit-fallback)特性。因此，在图模式中使用时，需要将`context`中的`jit_syntax_level`设置为`LAX`级别（2.1版本默认为此级别，无需修改）。此外，`save`语法的限制与该特性限制相同。目前已知的主要限制如下：
>   - `data`参数不支持传入函数返回值或表达式，例如`ts.save(file, func(x))`或`ts.save(file, x + 1)`可能会导致未定义行为。您可以使用临时变量保存中间结果，然后调用`save`函数来规避此问题，例如`t = func(x);ts.save(file, t)`。
>   - `file`参数对于全局变量的支持不完善，它只能获取全局变量在图编译完成后的值，无法获取在运行过程中修改的值；

### 样例

#### MindSpore动态图和静态图Ascend硬件平台GE流程使用方法

```python
import tempfile
from pathlib import Path

import troubleshooter as ts
import mindspore as ms
from mindspore import nn, Tensor

class NetWorkSave(nn.Cell):
    def __init__(self, file):
        super(NetWorkSave, self).__init__()
        self.file = file

    def construct(self, x):
        ts.save(self.file, x)
        return x

x1 = Tensor(-0.5962, ms.float32)
x2 = Tensor(0.4985, ms.float32)
dir = tempfile.TemporaryDirectory(prefix="save")
path = Path(dir.name)
net = NetWorkSave(str(path / "ms_tensor"))

# 文件会自动编号
out1 = net(x1)
# 0_ms_tensor.npy

out2 = net(x2)
# 1_ms_tensor.npy

out3 = net([x1, x2])
# 2_ms_tensor.0.npy
# 3_ms_tensor.1.npy

out4 = net({"x1": x1, "x2":x2})
# 4_ms_tensor.x1.npy
# 5_ms_tensor.x2.npy
```

#### MindSpore静态图Ascend硬件平台VM流程使用方法

VM流程下静态图模式数据保存有3个步骤。首先配置Print算子数据落盘，之后在网络中使用`ts.save`对数据进行保存，最后对落盘文件解析获得npy文件。

> **限制**
>
> - VM流程下，如果保存的数据过多（短时间内保存数据超过128条），Print算子可能丢失数据。

1. 配置print落盘

    1.1 单卡场景

    context直接设置中设置`print_file_path`保存路径。

    ```python
    ms.set_context(mode=ms.GRAPH_MODE, print_file_path=f'print_file_{rank_id}')
    ```

    1.2 多卡场景

    多卡场景下，**需要在计算卡通信初始化完成之前，设置print落盘环境变量**。

    多卡需要配置不同的落盘路径，可以在启动脚本中设置环境变量`RANK_ID`，在python脚本中获取配置到路径中（**路径相同时，多卡的数据将会错误的存储到一个文件中**）

    ```bash
    export RANK_ID=xx
    python xxx
    ```

    ```python
    import mindspore as ms
    
    # 获取环境变量，用来多卡保存在不同的文件中
    rank_id = os.getenv('RANK_ID')
    ms.set_context(mode=ms.GRAPH_MODE, print_file_path=f'print_file_{rank_id}')
    ```

2. 数据保存

    在数据保存时，设置`use_print`为True。

    ```python
    import troubleshooter as ts
    
    ...
    
    ts.save(file, data, use_print=True)
    ```

3. 数据解析

    在数据落盘后，需要对落盘的数据进行解析，获取npy文件，file为print落盘的文件，`output_dir`为转换后的npy保存的文件路径。

    ```python
    import troubleshooter as ts
    
    ts.save_convert(file='xx', output_dir='xxx')
    ```

#### PyTorch使用方法

```python
import tempfile
from pathlib import Path

import troubleshooter as ts
import torch
x1 = torch.tensor([-0.5962, 0.3123], dtype=torch.float32)
x2 = torch.tensor([[0.4985],[0.4323]], dtype=torch.float32)
dir = tempfile.TemporaryDirectory(prefix="save")
path = Path(dir.name)
file = str(path / "torch_tensor")

ts.save(file, x1)
# 0_torch_tensor.npy

ts.save(file, x2)
# 1_torch_tensor.npy

ts.save(file, {"x1":x1, "x2":x2})
# 2_torch_tensor.x1.npy
# 3_torch_tensor.x2.npy

ts.save(file, {"x1":x1, "x2":x2}, suffix="torch")
# 4_torch_tensor.x1_torch.npy
# 5_torch_tensor.x2_torch.npy
```
