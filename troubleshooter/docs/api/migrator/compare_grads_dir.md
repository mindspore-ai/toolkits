## troubleshooter.migrator.compare_grads_dir

> troubleshooter.migrator.compare_grads_dir(orig_dir, target_dir, rtol=1e-4, atol=1e-4, equal_nan=False, compare_shape=True, output_file=None)

批量对比两个目录下使用[save](api/save.md)保存梯度得到的npy文件。

使用[troubleshooter.migrator.get_name_map_list_by_shape_edit_distance](./get_name_map_list.md#troubleshootermigratorget_name_map_list_by_shape_edit_distance)规则获取名称映射规则。

和compare_npy_dir类似，同样会计算`numpy.allclose`、`allclose`达标比例、余弦相似度、差异值的 $mean$ / $max$ 统计量等信息，除此之外，默认会显示shape信息。

> **说明：**
>
> 1. 目前MindSpore获取的梯度不包含名称信息，在一些情况下，两边的网络结构可能不完全相同，按照顺序匹配会导致很多文件匹配失败。梯度比较的匹配策略是根据shape计算最小[编辑距离](https://baike.baidu.com/item/%E7%BC%96%E8%BE%91%E8%B7%9D%E7%A6%BB/8010193)，其中删除、插入代价为1，替换代价为5。如果需要调整代价可以直接调用`get_name_map_list_by_shape_edit_distance`函数，在获取`name_map_list`后直接传入`compare_grads_dir`。
>
> 2. torch的梯度需要通过`ts.widget.get_pt_grads`获取，如样例所示。

### 参数

- orig_dir(`str`): 需要对比的npy文件所在的目录。
- target_dir(`str`): 目标数据所在的目录。
- rtol(`float`): 相对误差，默认值为`1e-4`，内部调用`numpy.allclose`的参数。
- atol(`float`): 绝对误差，默认值为`1e-4`，内部调用`numpy.allclose`的参数。
- equal_nan(`bool`): 是否将nan视为相等，默认值为 `False`，内部调用`numpy.allclose`的参数。
- compare_shape(`bool`): 是否比较shape信息，默认值`True`。
- output_file(`str`): 比较结果导出为csv文件的路径，默认值`None`。

### 样例

```python
import os
import numpy as np
import troubleshooter as ts
import torch
import mindspore as ms
import tempfile
from pathlib import Path
class PtSimpleNet(torch.nn.Module):
    def __init__(self):
        super(PtSimpleNet, self).__init__()
        self.fc = torch.nn.Linear(10, 5)
        self.bn = torch.nn.BatchNorm1d(5)
    def forward(self, x):
        return self.bn(self.fc(x))

class MsSimpleNet(ms.nn.Cell):
    def __init__(self):
        super(MsSimpleNet, self).__init__()
        self.fc = ms.nn.Dense(10, 5)
    def construct(self, x):
        return self.fc(x)

pt_dir = tempfile.TemporaryDirectory(prefix="pt_")
ms_dir = tempfile.TemporaryDirectory(prefix="ms_")
pt_outpath = Path(pt_dir.name)
ms_outpath = Path(ms_dir.name)
inputs = np.random.randn(32, 10).astype(np.float32)
targets = np.random.randn(32, 5).astype(np.float32)

pt_net = PtSimpleNet()
pt_criterion = torch.nn.MSELoss()
pt_optimizer = torch.optim.SGD(pt_net.parameters(), lr=0.01)
pt_outputs = pt_net(torch.tensor(inputs))
pt_loss = pt_criterion(pt_outputs, torch.tensor(targets))
pt_optimizer.zero_grad()
pt_loss.backward()
# use ts.widget.get_pt_grads get torch grads
pt_grads = ts.widget.get_pt_grads(pt_net)
ts.save(str(pt_outpath / "torch_grads"), pt_grads)

ms_net = MsSimpleNet()
ms_loss_fn = ms.nn.MSELoss()

def forward_fn(inputs, targets):
    out = ms_net(inputs)
    loss = ms_loss_fn(out, targets)
    return loss

grad_fn = ms.value_and_grad(forward_fn, None, ms_net.trainable_params())
ms_loss, ms_grads = grad_fn(ms.Tensor(inputs), ms.Tensor(targets))
ts.save(str(ms_outpath / "ms_grads"), ms_grads)
ts.migrator.compare_grads_dir(pt_outpath, ms_outpath)
```

### 结果：

```bash
2023-11-18 17:31:45,065 - troubleshooter.log - WARNING - [*User Warning*] The number of files or their corresponding shapes are inconsistent, and some files may not be correctly mapped.
2023-11-18 17:31:45,066 - troubleshooter.log - WARNING - [*User Attention*] The compare directory information:
 The orig dir: /tmp/pt_9xatcvr0 
 The target dir: /tmp/ms_kfaide32
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:07<00:00,  1.92s/it]
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                    The list of comparison results                                                                   |
+-------------------------------+-------------------+---------------+-----------------+--------------------+-------------------+-------------------+------------------+
|        orig array name        | target array name | shape of orig | shape of target | result of allclose | ratio of allclose | cosine similarity | mean & max diffs |
+-------------------------------+-------------------+---------------+-----------------+--------------------+-------------------+-------------------+------------------+
| 0_torch_grads_fc.weight_0.npy |  1_ms_grads_0.npy |    (5, 10)    |     (5, 10)     |       False        |       0.00%       |      0.52350      | 0.10281, 0.34079 |
|  0_torch_grads_fc.bias_1.npy  |  1_ms_grads_1.npy |      (5,)     |       (5,)      |       False        |       0.00%       |      0.17462      | 0.17567, 0.22836 |
| 0_torch_grads_bn.weight_2.npy |        None       |      (5,)     |       None      |       False        |       0.00%       |        nan        |     nan, nan     |
|  0_torch_grads_bn.bias_3.npy  |        None       |      (5,)     |       None      |       False        |       0.00%       |        nan        |     nan, nan     |
+-------------------------------+-------------------+---------------+-----------------+--------------------+-------------------+-------------------+------------------+
```

