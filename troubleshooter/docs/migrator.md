# 网络迁移&调试-使用迁移功能快速迁移网络

## 应用场景1：pth到ckpt权重自动转换
用户需要从pytorch网络迁移到MindSpore网络时，需要进行权重迁移，因为MindSpore的权重名称与Pytorch有差异，需要使用权重迁移工具进行权重自动化迁移。
### 结果展示
显示转换后的ckpt保持路径与名称，并给出详细的转换信息。

![avatar](images/18h52_36.png)
### 如何使用1-网络结构完全一致，权重自动转换
在MindSpore迁移后的网络结构与pytorch网络结构完全一致时，pth到ckpt转换，仅有权重名称差异，则可以通过如下方法完成权重的自动转换。
```python
import troubleshooter as ts

torch_net = resnet50(num_classes=10)
pth_path="./resnet.pth"

"""
pt_model：pytorch网络实例。支持模型（例如：torch.save(torch_net, "torch_net.pth") ）和
参数（例如：torch.save(torch_net.state_dict(), "torch_net.pth"), 两种形式保存后的pth文件
的自动加载。如果保存的pth文件内容经过定制，不能进行自动加载，可使用"pth_para_dict"直接传入加载
并解析后的权重参数字典；
weight_map_save_path：转换后的权重映射表路径；
print_map: 是否打印映射表。
"""
ts.migrator.get_weight_map(pt_model=torch_net,
                           weight_map_save_path="/tmp/torch_net_map.json",
                           print_map=True)
"""
weight_map_path: get_weight_map生成的权重映射表路径；
pt_file_path: pytorch的pth文件路径；
ms_file_save_path: 转换后的MindSpore的ckpt文件路径。
"""
ts.migrator.convert_weight(weight_map_path="/tmp/torch_net_map.json",
                           pt_file_path="/tmp/torch_net.pth",
                           ms_file_save_path='/tmp/convert_resnet.ckpt')
```
### 如何使用2-网络结构有一定差异，需要定制权重名称前缀

```python
import troubleshooter as ts

# pytorch的resnet50网络
torch_net = resnet50(num_classes=10)
pth_path="./resnet.pth"

# weight_name_prefix：需要添加的权重前缀
ts.migrator.get_weight_map(pt_model=torch_net,
                           weight_map_save_path="/tmp/torch_net_map.json",
                           weight_name_prefix='uvp',
                           print_map=True)
# 调用转换接口
ts.migrator.convert_weight(weight_map_path="/tmp/torch_net_map.json",
                           pt_file_path="/tmp/torch_net.pth",
                           ms_file_save_path='/tmp/convert_resnet.ckpt')
```

### 如何使用3-网络结构有一定差异，需要对权重名称做复杂的定制转换
在MindSpore迁移后的网络结构与pytorch网络结构不完全一致时，需要用户手工定义转换规则，此时工具提供了定制接口，满足此种场景下用户的定制诉求。

```python
import troubleshooter as ts

def custorm_weight_name(weight_name_map):
    prefix='.custorm.'
    custorm_name_map = {}
    for key, value in weight_name_map.items():
        index = value.find(".")
        value = value[0:index] + prefix + value[index+1:]
        print(key, ":", value)
        custorm_name_map[key] = str(value)
    return custorm_name_map

# pytorch的resnet50网络
torch_net = resnet50(num_classes=10)
pth_path="./resnet.pth"

"""
custom_name_func: 可封装定制函数，例如：custorm_weight_name，完成映射关系的定制
"""
ts.migrator.get_weight_map(pt_model=torch_net,
                           weight_map_save_path="/tmp/torch_net_map.json",
                           custom_name_func=custorm_weight_name,
                           print_map=True)

# 调用转换接口
ts.migrator.convert_weight(weight_map_path="/tmp/torch_net_map.json",
                           pt_file_path="/tmp/torch_net.pth",
                           ms_file_save_path='/tmp/convert_resnet.ckpt')

# 执行结果：根据定制所有参数名称增加一个层custorm ，执行后举例: features.Linear_mm.weight 参数名称将转换为
# features.custorm.Linear_mm.weight
```


## 应用场景2：将转换后的ckpt与MindSpore网络生成的ckpt进行对比
我们写完MindSpore网络后，就可以保存一个ckpt。我们可以将网络生产的ckpt与转换的ckpt（用权重转换工具从pth转换过来的ckpt）进行对比，
以起到验证网络结构是否正确等目的。当前仅支持名称与shape的比对。
### 结果展示
![avatar](images/19h51_37.png)
### 如何使用1

```python
import troubleshooter as ts

weight_map_path = "./torch_net_map.json"
pth_path = "./resnet.pth"
ms_path = "./resnet.ckpt"

"""
weight_map_path: get_weight_map生成的权重映射表路径；
pt_file_path: pytorch的pth文件路径；
ms_file_path: 用户编写网络保存的ckpt文件路径。
"""
ts.migrator.compare_pth_and_ckpt(weight_map_path, pth_path, ms_path)
```

## 应用场景3：保存tensor
在网络迁移精度问题排查时，需要对网络中的数据进行保存。`troubleshooter`提供了支持`mindspore`和`pytorch`的统一数据保存接口，并支持文件自动编号功能。

### 接口定义

```python
ts.save(file:str, data:Union(Tensor, list[Tensor], tuple[Tensor], dict[str, Tensor], auto_id=True, suffix=None))
```
- file: 文件名路径。当`file`为`None`或`''`时，文件名会自动设置为`tensor_(shape)`，文件路径为当前路径。
- data: 数据，支持保存`Tensor`（包括`mindspore.Tensor`和`pytorch.tensor`），以及`Tensor`构成的`list/tuple/dict`。当为`list/tuple`类型时，会按照顺序添加编号；当为`dict`类型时，文件名中会添加`key`。
- auto_id: 自动编号，默认值为`True`。当为`True`时，保存时会自动为文件添加全局编号，编号从0开始。
- suffix: 文件名后缀，默认值为`None`。

**文件保存格式**

存储的文件名称为 `{id}_name_{idx/key}_{suffix}.npy`

### 如何使用

**支持mindspore动态图和静态图**

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

**支持pytorch**

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

## 应用场景4：比较两组tensor值(npy文件)是否相等
进行网络迁移精度问题排查等场景，需要获取网络中的tensor值进行比较。一般我们将tensor保存成npy进行手工比较，此功能提供了批量对比两个目录下名称可以
映射的npy值的接口。
### 结果展示
![avatar](images/20h21_53.png)
### 如何使用1-两个目录下名称可自动映射

```python
import troubleshooter as ts

ta = "/mnt/d/06_project/troubleshooter/troubleshooter/tests/diff_handler/ta"
tb = "/mnt/d/06_project/troubleshooter/troubleshooter/tests/diff_handler/tb"
# 比较ta与tb目录下名称可以映射的npy文件的值
ts.migrator.compare_npy_dir(ta, tb)
```

### 如何使用2-两个目录下名称不能完全映射，需要手工调整

```python
import troubleshooter as ts

ta = "/mnt/d/06_project/troubleshooter/troubleshooter/tests/diff_handler/ta"
tb = "/mnt/d/06_project/troubleshooter/troubleshooter/tests/diff_handler/tb"
# 可以通过如下接口获取名称映射列表，对npy文件名称映射进行调整
name_list = ts.migrator.get_filename_map_list(ta, tb)
# 通过自定定义一个函数进行list的修改，例如：custom_fun(name_list)
name_list = custom_fun(name_list)
# 将调整后的名称传入比较接口
ts.migrator.compare_npy_dir(name_map_list=name_list)
```

## 应用场景5：比较mindspore和pytorch网络输出是否一致

在进行网络迁移时，由于大多数网络是使用pytorch搭建的，在迁移到mindspore过程中，我们需要比较mindspore和pytorch网络输出结果是否一致。此功能实现对比mindspore和pytorch的输出结果。

### 接口参数

| 参数         | 类型                      | 说明                                                         |
| ------------ | ------------------------- | ------------------------------------------------------------ |
| ms_net       | `mindspore.nn.Cell`         | mindspore模型实例                                            |
| pt_net       | `torch.nn.Module`           | torch模型实例                                                |
| input_data   | 单输入：`Union(tuple[torch.tensor], tuple[mindspore.Tensor], tuple[numpy.ndarray], tuple[str])`；多输入：`list[Union(tuple[torch.tensor], tuple[mindspore.Tensor], tuple[numpy.ndarray], tuple[str])]`|模型输入。模型输入支持`torch.Tensor`, `mindspore.Tensor`, `np.ndarray`以及`str`，每个`tuple`中包含一个模型输入；当用户想要同时验证多组数据时，请使用一个列表存放所有输入。|
|auto_input_data|单输入：`tuple[tuple[numpy.shape, numpy.dtype]]`；多输入：`{'input': tuple[tuple[numpy.shape, numpy.dtype]], 'num':int}`|默认为`None`，为了方便用户快速验证。用户可以不输入`input_data`，而是输入`auto_input_data`，`auto_input_data`每一个元素为模型输入的`shape`，如果需要使用多次测试，可以传入一个字典，字典的键为`'input'`和`'num'`，分别表示每次的输入以及输入个数|

### 如何使用

可以参考troubleshooter/tests/diff_handler/test_netdifffinder.py中的使用方法，或者下面的使用方法：

```python
    pt_net = ConstTorch()
    ms_net = ConstMS()
    diff_finder = ts.migrator.NetDifferenceFinder(
        pt_net=pt_net,
        ms_net=ms_net,
        auto_input=(((1, 12), np.float32), ))
    diff_finder.compare()
```
输出结果：

![网络输出对比结果展示](images/outputcompare.png)

