## troubleshooter.save_grad
>
> troubleshooter.save_grad(file:str, data:Tensor, suffix='backward', output_mode='npy')

`MindSpore`和`PyTorch`的统一反向数据保存接口。

### 参数

- file(str): 文件名路径。为避免同名文件覆盖，保存的文件名称会自动添加前缀，前缀从0开始，按照执行顺序递增。
- data(Tensor): 数据，支持保存`Tensor`对应的反向梯度。
- suffix(str, 可选): 文件名后缀，默认值为`'backward'`。
- output_mode(str, 可选)：Tensor输出的模式，目前支持 `['npy','print']`，默认值`'npy'`。
    - `'npy'`模式会保存Tensor为numpy格式的npy文件，存储的文件名称为`[id]_name_[suffix].npy`，其中`id`为按照执行顺序自增的前缀；`name`为`file`中的文件名部分；`suffix`为指定的后缀，默认为`'backward'`；
    - `print`模式会将Tensor使用print输出到屏幕，输出内容依次为标识符`_TS_SAVE_NAME:`、`name`与`Tensor`。输出的`name`与`'npy'`模式类似，但不包含前缀`id`和文件路径，只包含文件名。print模式下，MindSpore Ascend平台图模式下支持配置context中的[print_file_path](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.print_.html)使Tensor**完整输出到文件**，输出的文件可以使用[save_convert](./widget/save_convert.md)解析为npy文件。不同模式与MindSpore版本支持对应关系如下。

| output_mode | 版本   | device         | 备注                                                         |
| ----------- | ------ | -------------- | ------------------------------------------------------------ |
| npy         | 2.3    | Ascend         | 图模式下只支持Ascend，pynative下支持Ascend/GPU/CPU。         |
|             | 2.3前  | 不支持         |                                                              |
| print       | 无依赖 | Ascend/GPU/CPU | Ascend图模式下支持将print数据完整输出到文件，需要配置context中的[print_file_path](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.print_.html)，输出的文件可以使用[save_convert](save_convert.md)解析为npy文件。 |

### 返回

- Tensor，与输入数据相同。**返回值需要传递给下一个API的输入，以构成反向图从而保存反向数据**。

### 文件名称格式

存储的文件名称为 **[id]\_name\_[suffix].npy**。

- id为按照执行顺序递增的前缀；
- name为file中指定的文件名；
- suffix为指定的后缀，默认为`backward`。

> **Warning:**
>
> - 在MindSpore 2.3之前的版本中，save_grad只支持使用print输出。
>
> - MindSpore 中数据保存是异步处理的。当数据量过大或主进程退出过快时，可能出现数据丢失的问题，需要主动控制主进程销毁时间，例如使用sleep。
>
> - 当前MindSpore支持保存的最大数据为2GB，包含100字节左右的数据描述头，当Tensor大小超过2GB时需要切片后再保存。

### 样例

#### MindSpore使用方法

output_mode='npy'方式如下，output_mode='print'保存到文件与文件转换流程与save相同，请参考[save output_mode='print'使用方法](save.md#mindspore-output_modeprint使用方法)。

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
        x = self.dense(x)
        # save the gradient of the input of the dense layer
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
    # ['ms_ts_save/0_dense_backward.npy']
    print(np.load(path / "0_dense_backward.npy"))
    # [[-0.5  0.5]]
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
        x = self.dense(x)
        x = ts.save_grad(self.path + "dense", x)
        return x


if __name__ == "__main__":
    path = './torch_ts_save/'
    data = np.array([[0.2, 0.5, 0.2]], dtype=np.float32)
    label = np.array([[1, 0]], dtype=np.float32)

    net = NetWithSaveGrad(path)
    loss_fun = torch.nn.CrossEntropyLoss()

    out = net(torch.tensor(data))
    loss = loss_fun(out, torch.tensor(label))
    loss.backward()
    time.sleep(0.1)
    path = Path(path)
    files = sorted(map(str, path.glob("*.npy")))
    print(files)
    # ['torch_ts_save/0_dense_backward.npy']
    print(np.load(path / "0_dense_backward.npy"))
    # [[-0.5  0.5]]
```
