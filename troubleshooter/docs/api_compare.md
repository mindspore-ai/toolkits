# API级别网络结果自动比较

## 应用场景

PyTorch 网络迁移到 MindSpore，以及 MindSpore 不同后端/版本迁移时，容易产生精度问题。正向问题往往需要二分保存中间变量，反向问题定位更是困难。API级别自动化比较功能实现了PyTorch 和 MindSpore 网络正、反向 API 级别(MindSpore:nn/ops/Tensor, PyTorch:nn/funcitonal/torch.xx/Tensor)输入、输出数据的自动保存以及对比，可以极大提高迁移效率。
同时支持 PyTorch 网络迁移到 MindTorch 调试时的精度对比，具体可参考[MindTorch调试调优指南-基于troubleshooter工具进行精度比较](https://openi.pcl.ac.cn/OpenI/MSAdapter/src/branch/master/doc/readthedocs/source_zh/docs/Tuning_Accuracy.md)

> 注意：
>
> 目前api_dump相关功能只支持MindSpore的PYNATIVE模式，暂不支持并行场景。

## 如何使用

精度对比时，首先需要保证三个条件，即需要满足随机性固定并相同、输入数据样本一致、初始化权重一致。具体可参考[网络精度对比三个条件](https://mindspore.cn/docs/zh-CN/master/migration_guide/migrator_with_tools.html#%E7%BD%91%E7%BB%9C%E6%AF%94%E5%AF%B9%E4%B8%89%E4%B8%AA%E5%9F%BA%E6%9C%AC%E6%9D%A1%E4%BB%B6)，以下示例中只简要说明。

> 注意：
>
> 为保证随机性相同，针对dropout类随机接口，api_dump会自动将p置为0，即输出数据和输入数据相同。

在满足三个条件后，就可以进行数据保存以及数据对比。

以下教程针对典型场景进行介绍，此外还可以实现对API的类型、范围、数据类型等选择性dump，具体可查看详细接口文档[api dump](./api/migrator/api_dump.md)。

### 数据保存

数据保存共有两个主要步骤。

1. 初始化dump功能。

    调用`api_dump_init`对MindSpore网络`Cell`/PyTorch网络`Module`执行初始化，指定输出路径，如果需要保存反向数据，需要指定`retain_backward`为True，详细请参考[api_dump_init](./api/migrator/api_dump.md#troubleshootermigratorapi_dump_init)。

2. 设置dump范围。

    在需要进行精度定位的代码上下调用`api_dump_start`和`api_dump_stop`。`api_dump_start`还可以指定dump的接口名、数据种类等，详细请参考[api_dump_start](./api/migrator/api_dump.md#troubleshootermigratorapi_dump_start)。

以下分别为PyTorch、MindSpore固定随机性、数据保存的完整代码示例。

**PyTorch 网络 dump**

```python
import troubleshooter as ts

from pathlib import Path

import numpy as np
import torch
from torch import nn, optim


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = nn.Conv2d(3, 5, kernel_size=3, stride=3, padding=0)
        self.bn = nn.BatchNorm2d(5)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(15000, 10)

    def forward(self, x):
        x = self.conv(x)
        x = torch.clip(x, 0.2, 0.5)
        x = self.bn(x)
        x = self.relu(x)
        x = x.reshape(1, -1)
        x = self.linear(x)
        x = self.relu(x)
        return x


def generate_data():
    data_path = Path('test_data')
    data_path.mkdir(exist_ok=True)
    np.save(data_path / 'label.npy',
            np.random.randn(1, 10).astype(np.float32))
    np.save(data_path / 'data.npy',
            np.random.randn(1, 3, 90, 300).astype(np.float32))
    return data_path


def train_one_step_torch(data_path):
    # 1. 固定随机性
    ts.fix_random()

    # 2. 创建训练网络
    net = SimpleNet()
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad()
    criterion = nn.MSELoss()

    # 3. 统一数据样本
    data = torch.tensor(np.load(data_path/'data.npy'))
    label = torch.tensor(np.load(data_path/'label.npy'))

    # 4. 保存权重和转换映射，用于在MindSpore加载
    info_path = 'pt_net_info'
    ts.migrator.save_net_and_weight_params(net, path=info_path)

    # 5. 设置dump网络
    ts.migrator.api_dump_init(
        net, output_path="torch_dump", retain_backward=True)

    # 6. 在迭代开始时开启dump
    ts.migrator.api_dump_start()

    # 7. 执行训练流程
    pred = net(data)
    loss = criterion(pred, label)
    loss.backward()

    # 8. 在优化器更新前关闭dump
    ts.migrator.api_dump_stop()

    # 9. 执行优化器更新
    optimizer.step()

# 生成数据样本，确保PyTorch和MindSpore网络输入一致
data_path = generate_data()

train_one_step_torch(data_path)

```

**MindSpore 网络 dump**

```python
import troubleshooter as ts

from pathlib import Path

import mindspore as ms
import numpy as np
from mindspore import nn, ops


class SimpleNet(nn.Cell):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = nn.Conv2d(3, 5, kernel_size=3, stride=3,
                              padding=0, pad_mode="pad", has_bias=True)
        self.bn = nn.BatchNorm2d(5)
        self.relu = nn.ReLU()
        self.linear = nn.Dense(15000, 10)

    def construct(self, x):
        x = self.conv(x)
        x = ops.clip(x, ms.Tensor(0.2, ms.float32), ms.Tensor(0.5, ms.float32))
        x = self.bn(x)
        x = self.relu(x)
        x = x.reshape(1, -1)
        x = self.linear(x)
        x = self.relu(x)
        return x


def generate_data():
    data_path = Path('test_data')
    data_path.mkdir(exist_ok=True)
    np.save(data_path / 'label.npy',
            np.random.randn(1, 10).astype(np.float32))
    np.save(data_path / 'data.npy',
            np.random.randn(1, 3, 90, 300).astype(np.float32))
    return data_path


def train_one_step_torch(data_path):
    # 1. 固定随机性
    ts.fix_random()

    # 2. 创建训练网络
    net = SimpleNet()
    net.set_train()
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
    criterion = nn.MSELoss()

    def forward_fn(data, label):
        out = net(data)
        loss = criterion(out, label)
        return loss
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    # 3. 统一数据样本
    data = ms.Tensor(np.load(data_path/'data.npy'))
    label = ms.Tensor(np.load(data_path/'label.npy'))

    # 4. 保存权重和转换映射，用于在MindSpore加载
    info_path = Path('pt_net_info')
    ts.migrator.convert_weight_and_load(weight_map_path=info_path/"torch_net_map.json",
                                        pt_file_path=info_path/"torch_troubleshooter_create.pth",
                                        net=net)

    # 5. 设置dump网络
    ts.migrator.api_dump_init(net, output_path="ms_dump", retain_backward=True)

    # 6. 在迭代开始时开启dump
    ts.migrator.api_dump_start()

    # 7. 执行训练流程
    loss, grads = grad_fn(data, label)

    # 8. 在优化器更新前关闭dump
    ts.migrator.api_dump_stop()

    # 9. 执行优化器更新
    optimizer(grads)

# 使用与PyTorch相同的数据作为输入
data_path = Path('test_data')

train_one_step_torch(data_path)

```

在执行完dump后，会在指定的目录下生成堆栈信息、执行信息、`npy`文件等数据，详细的数据格式说明请参考[数据格式](./api/migrator/api_dump.md#数据格式)。

### 数据对比

比较时是需要输入dump路径，即可自动识别是MindSpore还是PyTorch网络dump的数据，实现PyTorch或MindSpore迁移的对比。`api_dump_compare`还可以设置比较的精度、未匹配项的显示方式等，具体请参考[api_dump_compare](./api/migrator/api_dump.md#troubleshootermigratorapi_dump_compare)。

```python
import troubleshooter as ts

# 根据dump得到的数据进行比对，指定对比结果输出到output_path目录下
ts.migrator.api_dump_compare('ms_dump', 'torch_dump', output_path='compare_result')

```

对比结果一共分为三个部分，第一部分是 API 映射关系，第二部分是正向对比结果，如果对反向进行了 dump，还会有第三部分反向的对比结果。会生成 ts_api_mapping.csv（API 映射文件）、 ts_api_forward_compare.csv（正向比对结果）、ts_api_backward_compare.csv（反向比对结果）。

**API 映射关系**

The APIs mapping results of the two frameworks

| ORIGIN NET (mindspore) |  TARGET NET (pytorch) |
|------------------------|-----------------------|
|      NN_Conv2d_0       |      NN_Conv2d_0      |
|   Functional_clip_0    |      Torch_clip_0     |
|    NN_BatchNorm2d_0    |    NN_BatchNorm2d_0   |
|       NN_ReLU_0        |       NN_ReLU_0       |
|    Tensor_reshape_0    |    Tensor_reshape_0   |
|       NN_Dense_0       |      NN_Linear_0      |
|       NN_ReLU_1        |       NN_ReLU_1       |

**正向结果对比**

The forward comparison results

正向对比结果按照执行的顺序进行比较，差异项分别为`numpy.allclose`、`allclose`达标比例、余弦相似度、差异值的 $mean$ / $max$ 统计量等信息。

|         ORIGIN NET (mindspore)        |         TARGET NET (pytorch)         |  shape of orig  | shape of target | result of allclose | ratio of allclose | cosine similarity | mean & max diffs |
|---------------------------------------|--------------------------------------|-----------------|-----------------|--------------------|-------------------|-------------------|------------------|
|    NN_Conv2d_0_forward_input.0.npy    |   NN_Conv2d_0_forward_input.0.npy    | (1, 3, 90, 300) | (1, 3, 90, 300) |        True        |      100.00%      |      1.00000      | 0.00000, 0.00000 |
|     NN_Conv2d_0_forward_output.npy    |    NN_Conv2d_0_forward_output.npy    | (1, 5, 30, 100) | (1, 5, 30, 100) |        True        |      100.00%      |      1.00000      | 0.00000, 0.00000 |
| Functional_clip_0_forward_input.0.npy |   Torch_clip_0_forward_input.0.npy   | (1, 5, 30, 100) | (1, 5, 30, 100) |        True        |      100.00%      |      1.00000      | 0.00000, 0.00000 |
|  Functional_clip_0_forward_output.npy |   Torch_clip_0_forward_output.npy    | (1, 5, 30, 100) | (1, 5, 30, 100) |        True        |      100.00%      |      0.99999      | 0.00000, 0.00000 |
|  NN_BatchNorm2d_0_forward_input.0.npy | NN_BatchNorm2d_0_forward_input.0.npy | (1, 5, 30, 100) | (1, 5, 30, 100) |        True        |      100.00%      |      0.99999      | 0.00000, 0.00000 |
|  NN_BatchNorm2d_0_forward_output.npy  | NN_BatchNorm2d_0_forward_output.npy  | (1, 5, 30, 100) | (1, 5, 30, 100) |        True        |      100.00%      |      1.00000      | 0.00000, 0.00001 |
|     NN_ReLU_0_forward_input.0.npy     |    NN_ReLU_0_forward_input.0.npy     | (1, 5, 30, 100) | (1, 5, 30, 100) |        True        |      100.00%      |      1.00000      | 0.00000, 0.00001 |
|      NN_ReLU_0_forward_output.npy     |     NN_ReLU_0_forward_output.npy     | (1, 5, 30, 100) | (1, 5, 30, 100) |        True        |      100.00%      |      0.99999      | 0.00001, 0.00001 |
|  Tensor_reshape_0_forward_input.0.npy | Tensor_reshape_0_forward_input.0.npy | (1, 5, 30, 100) | (1, 5, 30, 100) |        True        |      100.00%      |      0.99999      | 0.00001, 0.00001 |
|  Tensor_reshape_0_forward_output.npy  | Tensor_reshape_0_forward_output.npy  |    (1, 15000)   |    (1, 15000)   |        True        |      100.00%      |      0.99999      | 0.00001, 0.00001 |
|     NN_Dense_0_forward_input.0.npy    |   NN_Linear_0_forward_input.0.npy    |    (1, 15000)   |    (1, 15000)   |        True        |      100.00%      |      0.99999      | 0.00001, 0.00001 |
|     NN_Dense_0_forward_output.npy     |    NN_Linear_0_forward_output.npy    |     (1, 10)     |     (1, 10)     |        True        |      100.00%      |      0.99999      | 0.00000, 0.00000 |
|     NN_ReLU_1_forward_input.0.npy     |    NN_ReLU_1_forward_input.0.npy     |     (1, 10)     |     (1, 10)     |        True        |      100.00%      |      0.99999      | 0.00000, 0.00000 |
|      NN_ReLU_1_forward_output.npy     |     NN_ReLU_1_forward_output.npy     |     (1, 10)     |     (1, 10)     |        True        |      100.00%      |      1.00000      | 0.00000, 0.00000 |

**反向对比结果**

The backward comparison results

反向结果显示为正向的逆序

|        ORIGIN NET (mindspore)        |         TARGET NET (pytorch)        |  shape of orig  | shape of target | result of allclose | ratio of allclose | cosine similarity | mean & max diffs |
| ------------------------------------ | -------------------------------------- | ------------- | --------------- | ------------------ | ----------------- | ----------------- | ---------------- |
|     NN_ReLU_1_backward_input.npy     |     NN_ReLU_1_backward_input.npy    |     (1, 10)     |     (1, 10)     |        True        |      100.00%      |      1.00000      | 0.00000, 0.00000 |
|    NN_Dense_0_backward_input.npy     |    NN_Linear_0_backward_input.npy   |     (1, 10)     |     (1, 10)     |        True        |      100.00%      |      1.00000      | 0.00000, 0.00000 |
| Tensor_reshape_0_backward_input.npy  | Tensor_reshape_0_backward_input.npy |    (1, 15000)   |    (1, 15000)   |        True        |      100.00%      |      1.00000      | 0.00000, 0.00000 |
|     NN_ReLU_0_backward_input.npy     |     NN_ReLU_0_backward_input.npy    | (1, 5, 30, 100) | (1, 5, 30, 100) |        True        |      100.00%      |      1.00000      | 0.00000, 0.00000 |
| NN_BatchNorm2d_0_backward_input.npy  | NN_BatchNorm2d_0_backward_input.npy | (1, 5, 30, 100) | (1, 5, 30, 100) |        True        |      100.00%      |      1.00000      | 0.00000, 0.00000 |
| Functional_clip_0_backward_input.npy |   Torch_clip_0_backward_input.npy   | (1, 5, 30, 100) | (1, 5, 30, 100) |        True        |      100.00%      |      1.00000      | 0.00000, 0.00000 |
|    NN_Conv2d_0_backward_input.npy    |    NN_Conv2d_0_backward_input.npy   | (1, 5, 30, 100) | (1, 5, 30, 100) |        True        |      100.00%      |      1.00000      | 0.00000, 0.00000 |

可以根据正反向对比结果，找到开始出现差异的api进行定位，减少二分的时间成本，提高定位效率。
