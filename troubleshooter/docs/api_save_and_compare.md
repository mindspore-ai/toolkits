# API级别数据保存与对比

## 应用场景

PyTorch 网络迁移到 MindSpore，以及 MindSpore 不同后端/版本迁移时，容易产生精度问题。

TroubleShooter提供了MindSpore与PyTroch统一的API级别正向数据保存接口[save](api/save.md)与反向梯度保存接口[save_grad](api/save_grad.md)（支持MindSpore图模式以及数据并行），可以直接把Tensor数据保存为numpy格式的npy文件；并且提供了批量数据对比接口[compare_npy_dir](api/migrator/compare_npy_dir.md)，可以批量对比两个目录下的npy文件。通过数据保存和对比两个功能的组合，可以极大提高精度定位效率。

> 注意：
>
> MindSpore 2.3之前的版本不支持save与save_grad直接保存npy，之前的版本中在Ascend图模式下可以借助print落盘方式进行保存，详情请查看[save中output_mode参数](api/save.md)。

## 如何使用

### 数据保存

以下用例使用save接口保存了MindSpore和Torch网络的输入、dense层的输出，使用save_grad保存了dense层反向输入的梯度。

在保存数据时可以添加文件名后缀，save接口默认无后缀，反向save_grad接口默认添加`'backward'`后缀，详细请参考[save的suffix参数](api/save.md)。

**需要注意的是，在保存梯度时，需要将save_grad的输出作为下一个API的输入，以构成反向图从而保存反向数据。**

#### MindSpore使用方法

```python
import troubleshooter as ts
import time
import numpy as np
import mindspore as ms
from mindspore.common.initializer import initializer, Zero

from pathlib import Path


class NetWithSaveGrad(ms.nn.Cell):
    def __init__(self, path):
        super(NetWithSaveGrad, self).__init__()
        self.dense = ms.nn.Dense(3, 2)
        self.apply(self._init_weights)
        self.path = path

    def _init_weights(self, cell):
        if isinstance(cell, ms.nn.Dense):
            cell.weight.set_data(
                initializer(Zero(), cell.weight.shape, cell.weight.dtype)
            )
            cell.bias.set_data(initializer(
                Zero(), cell.bias.shape, cell.bias.dtype))

    def construct(self, x):
        # save input
        ts.save(self.path + "input", x)
        x = self.dense(x)
        # save dense output
        ts.save(self.path + "dense", x)
        # save the gradient of the input of the dense layer
        # the output of the save_grad should be passed to the next API as input
        x = ts.save_grad(self.path + "dense", x)
        return x


if __name__ == "__main__":
    ms.set_context(mode=ms.PYNATIVE_MODE)
    path = './ms_ts_save/'
    data = np.array([[0.2, 0.5, 0.2]], dtype=np.float32)
    label = np.array([[1, 0]], dtype=np.float32)

    net = NetWithSaveGrad(path)
    loss_fn = ms.nn.CrossEntropyLoss()

    def forward_fn(data, label):
        logits = net(data)
        loss = loss_fn(logits, label)
        return loss, logits

    grad_fn = ms.grad(
        forward_fn,
        grad_position=None,
        weights=net.trainable_params(),
        has_aux=True,
        return_ids=True,
    )
    grads, _ = grad_fn(ms.Tensor(data), ms.Tensor(label))
    time.sleep(0.1)
    path = Path(path)
    files = sorted(map(str, path.glob("*.npy")))
    print(files)
    # ['ms_ts_save/0_input.npy', 'ms_ts_save/1_dense.npy', 'ms_ts_save/2_dense_backward.npy']

```

#### PyTorch使用方法

```python
import troubleshooter as ts
import time
import numpy as np
import torch

from pathlib import Path


class NetWithSaveGrad(torch.nn.Module):
    def __init__(self, path):
        super(NetWithSaveGrad, self).__init__()
        self.dense = torch.nn.Linear(3, 2)
        self.apply(self._init_weights)
        self.path = path

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.zero_()
            module.bias.data.zero_()

    def forward(self, x):
        # save input
        ts.save(self.path + "input", x)
        x = self.dense(x)
        # save dense output
        ts.save(self.path + "dense", x)
        # save the gradient of the input of the dense layer
        # the output of the save_grad should be passed to the next API as input
        x = ts.save_grad(self.path + "dense", x)
        return x


if __name__ == "__main__":
    path = './torch_ts_save/'
    data = np.array([[0.2, 0.5, 0.2]], dtype=np.float32)
    label = np.array([[1, 0]], dtype=np.float32)
    label_pt = np.array([0], dtype=np.float32)

    net = NetWithSaveGrad(path)
    loss_fun = torch.nn.CrossEntropyLoss()

    out = net(torch.tensor(data))
    loss = loss_fun(out, torch.tensor(label))
    loss.backward()
    time.sleep(0.1)
    path = Path(path)
    files = sorted(map(str, path.glob("*.npy")))
    print(files)
    # ['torch_ts_save/0_input.npy', 'torch_ts_save/1_dense.npy', 'torch_ts_save/2_dense_backward.npy']

```

### 数据对比

在保存之后可以使用[compare_npy_dir](./api/migrator/compare_npy_dir.md)进行数据比较，具体参数请参考[compare_npy_dir](./api/migrator/compare_npy_dir.md)。

```python
import troubleshooter as ts

ts.migrator.compare_npy_dir('ms_ts_save', 'torch_ts_save')

```

执行结果如下，比较结果分别为`numpy.allclose`、`allclose`达标比例、余弦相似度、差异值的 $mean$ / $max$ 统计量等信息。

```
2023-12-11 14:45:14,907 - troubleshooter.log - WARNING - [*User Attention*] The compare directory information:
 The orig dir: ms_ts_save 
 The target dir: torch_ts_save
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 702.60it/s]
+-----------------------------------------------------------------------------------------------------------------------------+
|                                                The list of comparison results                                               |
+----------------------+----------------------+--------------------+-------------------+-------------------+------------------+
|   orig array name    |  target array name   | result of allclose | ratio of allclose | cosine similarity | mean & max diffs |
+----------------------+----------------------+--------------------+-------------------+-------------------+------------------+
|     0_input.npy      |     0_input.npy      |        True        |      100.00%      |      1.00000      | 0.00000, 0.00000 |
|     1_dense.npy      |     1_dense.npy      |        True        |      100.00%      |      1.00000      | 0.00000, 0.00000 |
| 2_dense_backward.npy | 2_dense_backward.npy |        True        |      100.00%      |      1.00000      | 0.00000, 0.00000 |
+----------------------+----------------------+--------------------+-------------------+-------------------+------------------+
```

除此之外还可以指定allclose的rtol、atol、equal_nan，比较shape，设置输出文件等，具体参数请查看[compare_npy_dir](./api/migrator/compare_npy_dir.md)。

例如下面例子，allclose的rtol、atol设置为`1e-3`，euqal_nan设置为True，并比较两个数据的shape，并把最终的比较结果输出到'compare.csv'文件中。

```python
ts.migrator.compare_npy_dir('ms_ts_save', 'torch_ts_save', rtol=1e-3, atol=1e-3, equal_nan=True, compare_shape=True, output_file='./compare.csv')
```

执行结果会打印到屏幕，并且会输出到'compare.csv'文件中。

```
2023-12-11 15:15:10,903 - troubleshooter.log - WARNING - [*User Attention*] The compare directory information:
 The orig dir: ms_ts_save 
 The target dir: torch_ts_save
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 1101.83it/s]
+---------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                 The list of comparison results                                                                |
+----------------------+----------------------+---------------+-----------------+--------------------+-------------------+-------------------+------------------+
|   orig array name    |  target array name   | shape of orig | shape of target | result of allclose | ratio of allclose | cosine similarity | mean & max diffs |
+----------------------+----------------------+---------------+-----------------+--------------------+-------------------+-------------------+------------------+
|     0_input.npy      |     0_input.npy      |     (1, 3)    |      (1, 3)     |        True        |      100.00%      |      1.00000      | 0.00000, 0.00000 |
|     1_dense.npy      |     1_dense.npy      |     (1, 2)    |      (1, 2)     |        True        |      100.00%      |      1.00000      | 0.00000, 0.00000 |
| 2_dense_backward.npy | 2_dense_backward.npy |     (1, 2)    |      (1, 2)     |        True        |      100.00%      |      1.00000      | 0.00000, 0.00000 |
+----------------------+----------------------+---------------+-----------------+--------------------+-------------------+-------------------+------------------+
```
