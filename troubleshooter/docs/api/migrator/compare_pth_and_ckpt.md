## troubleshooter.migrator.compare_pth_and_ckpt

> troubleshooter.migrator.compare_pth_and_ckpt(weight_map_path, pt_file_path, ms_file_path, **kwargs)

用于比较PyTorch和MindSpore的结构和数值差异。

### 常用参数：

- weight_map_path(str)：通过get_weight_map函数生成的权重映射表路径。
- pt_file_path(str)：PyTorch的pth文件路径。
- ms_file_path(str)：MindSpore的ckpt的文件路径。

### kwargs参数：

- compare_value(bool)：是否进行数值比较，默认值为True。为True时，会分别输出shape和value两个差异分析表格。
- print_level(int)：日志等级，默认值为1。为0时不输出比较结果，为1时输出所有结果，为2时仅输出有差异的结果。
- rtol(float): 开启数值比较时的比较参数，相对误差，默认值为`1e-4`，内部调用`numpy.allclose`的参数。
- atol(float): 开启数值比较时的比较参数，绝对误差，默认值为`1e-4`，内部调用`numpy.allclose`的参数。
- equal_nan(bool)：开启数值比较时的比较参数，是否将nan视为相等，默认值为 `False`，内部调用`numpy.allclose`的参数。

### 样例：

```python
import troubleshooter as ts
import torch
import mindspore as ms
from mindspore import nn


class PTNet(torch.nn.Module):
    def __init__(self):
        super(PTNet, self).__init__()
        self.fc1 = torch.nn.Linear(2, 10)
        self.bn1 = torch.nn.BatchNorm1d(10)
        self.fc2 = torch.nn.Linear(10, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def construct(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

class MSNet(nn.Cell):
    def __init__(self):
        super(MSNet, self).__init__()
        self.fc1 = nn.Dense(3, 10)
        self.bn1 = nn.BatchNorm1d(10)
        self.fc2 = nn.Dense(10, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

pt_net = PTNet()
ms_net = MSNet()
ts.migrator.get_weight_map(pt_net=pt_net,
                           weight_map_save_path="torch_net_map.json")
torch.save(pt_net.state_dict(), "pt_net.pth")
ms.save_checkpoint(ms_net, "ms_net.ckpt")
ts.migrator.compare_pth_and_ckpt("torch_net_map.json", "pt_net.pth", "ms_net.ckpt", print_level=2)
```

![compare_pth_ckpt](../../images/compare_pth_ckpt.png)

