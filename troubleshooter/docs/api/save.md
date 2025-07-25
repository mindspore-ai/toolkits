## troubleshooter.save
>
> troubleshooter.save(file:str, data:Union(Tensor, list[Tensor], tuple[Tensor], dict[str, Tensor]), suffix=None，output_mode='npy')

`MindSpore`和`PyTorch`的统一数据保存接口。

### 参数

- file(str): 文件名路径。为避免同名文件覆盖，保存的文件名称会自动添加后缀，后缀从0开始，按照执行顺序递增。
- data(Union(Tensor, list[Tensor], tuple[Tensor], dict[str, Tensor])): 数据，支持保存`Tensor`（包括`mindspore.Tensor`和`pytorch.Tensor`），以及`Tensor`构成的`list/tuple/dict`。
- suffix(str, 可选): 文件名后缀，默认值为`None`。
- output_mode(str, 可选)：Tensor输出的模式，目前支持 `['npy','print']`，默认值`'npy'`。
    - `'npy'`模式会保存Tensor为numpy格式的npy文件，存储的文件名称为`name.[idx/key]_[suffix]_[dtype]_[id].npy`，其中`name`为`file`中的文件名部分；`dtype`为数据原始类型（bfloat16类型会转化为float32类型保存）；`id`为按照执行顺序自增的后缀；当数据为`list/tuple`类型时，会按照索引顺序添加`idx`，为`dict`类型时，文件名中会添加`key`，使用点分隔；`suffix`为指定的后缀，默认为空；
    - `print`模式会将Tensor使用print输出到屏幕，输出内容依次为标识符`_TS_SAVE_NAME:`、`name`与`Tensor`。输出的`name`与`'npy'`模式类似，但不包含后缀`id`和文件路径，只包含文件名。print模式下，MindSpore Ascend平台图模式下支持配置context中的[print_file_path](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.print_.html)使Tensor**完整输出到文件**，输出的文件可以使用[save_convert](./widget/save_convert.md)解析为npy文件。不同模式与MindSpore版本支持对应关系如下。

| output_mode | 版本   | device         | 备注                                                         |
| ----------- | ------ | -------------- | ------------------------------------------------------------ |
| npy         | 2.3    | Ascend         | 图模式下只支持Ascend，pynative下支持Ascend/GPU/CPU。         |
|             | 2.3前  | 不支持         |                                                              |
| print       | 无依赖 | Ascend/GPU/CPU | Ascend图模式下支持将print数据完整输出到文件，需要配置context中的[print_file_path](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.print_.html)，输出的文件可以使用[save_convert](save_convert.md)解析为npy文件。 |

> **Warning:**
>
> - 在Ascend平台上的Graph模式下，可以通过设置环境变量 MS_DUMP_SLICE_SIZE 和 MS_DUMP_WAIT_TIME 解决在输出大Tesnor或输出Tensor比较密集场景下算子执行失败的问题，详情请查看[环境变量](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/env_var_list.html)。
>
> - 在MindSpore 2.3之前的版本中，save只支持使用print输出。
>
> - 暂不支持在MindSpore自定义反向中保存数据。

### 样例

#### MindSpore output_mode='npy'使用方法

```python
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
path = Path('save_ms')
net = NetWorkSave(str(path / "ms_tensor"))

# 文件会自动编号
out1 = net(x1)
# save_ms/ms_tensor_float32_0.npy

out2 = net(x2)
# save_ms/ms_tensor_float32_1.npy

out3 = net([x1, x2])
# save_ms/ms_tensor.0_float32_2.npy
# save_ms/ms_tensor.1_float32_3.npy

out4 = net({"x1": x1, "x2":x2})
# save_ms/ms_tensor.x1_float32_4.npy
# save_ms/ms_tensor.x2_float32_5.npy
```

#### PyTorch使用方法

```python
from pathlib import Path

import troubleshooter as ts
import torch
x1 = torch.tensor([-0.5962, 0.3123], dtype=torch.float32)
x2 = torch.tensor([[0.4985],[0.4323]], dtype=torch.float32)
path = Path('save_pt')
file = str(path / "torch_tensor")

ts.save(file, x1)
# save_pt/torch_tensor_float32_0.npy

ts.save(file, x2)
# save_pt/torch_tensor_float32_1.npy

ts.save(file, {"x1":x1, "x2":x2})
# save_pt/torch_tensor.x1_float32_2.npy
# save_pt/torch_tensor.x2_float32_3.npy

ts.save(file, {"x1":x1, "x2":x2}, suffix="torch")
# save_pt/torch_tensor.x1_torch_float32_4.npy
# save_pt/torch_tensor.x2_torch_float32_5.npy
```

#### MindSpore output_mode='print'使用方法

output_mode='print'，默认会输出到屏幕。Ascend图模式下，MindSpore支持将print数据完整保存到文件，troubleshooter支持将此种print文件转换为npy文件，共有3个步骤。首先需要配置Print算子数据落盘路径，之后在网络中使用`ts.save`的`'print'`模式进行数据保存，最后对落盘文件解析获得npy文件，以下详细介绍。

> **限制**
>
> - 设置print_file_path后，数据只会保存在文件，不会输出到屏幕；
>
> - MindSpore 2.3 版本前，如果保存的数据过多（短时间内保存数据超过128条），Print算子可能丢失数据。

1. 配置print落盘

    1.1 单卡场景

    在context设置的[print_file_path](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.print_.html)使Print输出到文件。

    ```python
    ms.set_context(mode=ms.GRAPH_MODE, print_file_path=f'print_file')
    ```

    1.2 多卡场景

    多卡场景下，**需要在计算卡通信初始化完成之前，设置print落盘环境变量**。

    多卡需要配置不同的落盘路径，若为多进程启动且使用相对路径时，文件会保存在相应的文件夹下。如果使用绝对路径，可以在启动脚本中设置环境变量`RANK_ID`，在python脚本中获取配置到路径中（**路径相同时，多卡的数据将会错误的存储到一个文件中**）

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

    在数据保存时，设置`output_mode`为`'print'`模式。

    ```python
    import troubleshooter as ts
    
    ...
    
    ts.save(file, data, output_mode='print')
    ```

3. 数据解析

    在数据落盘后，需要对落盘的数据进行解析，获取npy文件，file为print落盘的文件，`output_path`为转换后的npy保存的文件路径。

    ```python
    import troubleshooter as ts
    
    ts.widget.save_convert(file='xx', output_path='xxx')
    ```
