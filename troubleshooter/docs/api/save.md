## troubleshooter.save
> troubleshooter.save(file:str, data:Union(Tensor, list[Tensor], tuple[Tensor], dict[str, Tensor], auto_id=True, suffix=None))

### 参数

- file: 文件名路径。当`file`为`None`或`''`时，文件名会自动设置为`tensor_(shape)`，文件路径为当前路径。
- data: 数据，支持保存`Tensor`（包括`mindspore.Tensor`和`pytorch.tensor`），以及`Tensor`构成的`list/tuple/dict`。当为`list/tuple`类型时，会按照顺序添加编号；当为`dict`类型时，文件名中会添加`key`。
- auto_id: 自动编号，默认值为`True`。当为`True`时，保存时会自动为文件添加全局编号，编号从0开始。
- suffix: 文件名后缀，默认值为`None`。

**文件保存格式**

存储的文件名称为 **{id}\_name\_{idx/key}\_{suffix}.npy`**

> **Warning:**
>
> - 在MindSpore 2.0版本中，save函数暂时不支持图模式。
>
> - 在2.1版本中，save函数支持MindSpore图模式，但实现依赖于[JIT Fallback](https://mindspore.cn/docs/zh-CN/master/design/dynamic_graph_and_static_graph.html#jit-fallback)特性。因此，在图模式中使用时，需要将`context`中的`jit_syntax_level`设置为`LAX`级别（2.1版本默认为此级别，无需修改）。此外，`save`语法的限制与该特性限制相同。目前已知的主要限制如下：
>   - `data`参数不支持传入函数返回值或表达式，例如`ts.save(file, func(x))`或`ts.save(file, x + 1)`可能会导致未定义行为。您可以使用临时变量保存中间结果，然后调用`save`函数来规避此问题，例如`t = func(x);ts.save(file, t)`。
>   - `file`参数对于全局变量的支持不完善，它只能获取全局变量在图编译完成后的值，无法获取在运行过程中修改的值；

### 用例

**支持MindSpore动态图和静态图**

```python
import os
import shutil

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
try:
    shutil.rmtree("/tmp/save/")
except FileNotFoundError:
    pass
os.makedirs("/tmp/save/")
net = NetWorkSave('/tmp/save/ms_tensor')

# 支持自动编号
out1 = net(x1)
# /tmp/save/0_ms_tensor.npy

out2 = net(x2)
# /tmp/save/1_ms_tensor.npy

out3 = net([x1, x2])
# /tmp/save/2_ms_tensor_0.npy
# /tmp/save/2_ms_tensor_1.npy

out4 = net({"x1": x1, "x2":x2})
# /tmp/save/3_ms_tensor_x1.npy
# /tmp/save/3_ms_tensor_x2.npy
```

**支持PyTorch**

```python
import os
import shutil

import troubleshooter as ts
import torch
x1 = torch.tensor([-0.5962, 0.3123], dtype=torch.float32)
x2 = torch.tensor([[0.4985],[0.4323]], dtype=torch.float32)

try:
    shutil.rmtree("/tmp/save/")
except FileNotFoundError:
    pass
os.makedirs("/tmp/save/")

file = '/tmp/save/torch_tensor'

ts.save(file, x1)
# /tmp/save/0_torch_tensor.npy
ts.save(file, x2)
# /tmp/save/1_torch_tensor.npy
ts.save(None, {"x1":x1, "x2":x2}, suffix="torch")
# ./2_tensor_(2,)_x1_torch.npy
# ./2_tensor_(2, 1)_x2_torch.npy
```
