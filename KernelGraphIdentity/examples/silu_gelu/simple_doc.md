# 执行序比对用例1-单算子差异比较

## 步骤一：执行用户代码生成执行需文件

```python
import os

import mindspore as ms
from mindspore import nn, ops, Tensor
import numpy as np

# 设置执行序环境变量
os.environ["MS_ALLOC_CONF"] = "memory_tracker:True" 
# 设置昇腾设备环境
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")  # 强制启用图模式加速[1,5]
ms.set_context(device_id=0)  # 指定第0号芯片

class AscendNet(nn.Cell):
    def __init__(self):
        super().__init__()
        # 定义5个算子
        self.conv = nn.Conv2d(3, 64, kernel_size=3, pad_mode='same')  # 卷积算子
        self.relu = ops.ReLU()                                      # 激活函数
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)           # 池化算子
        self.flatten = nn.Flatten()                                 # 展平算子
        self.fc = nn.Dense(64 * 16 * 16, 10)                            # 全连接层

    def construct(self, x):
        x = self.conv(x)        # 3通道→64通道特征提取
        x = self.relu(x)        # 非线性激活
        x = self.pool(x)        # 下采样降维
        x = self.flatten(x)     # 数据展平
        x = self.fc(x)          # 全连接分类
        return x

# 生成模拟数据（替代真实数据集）
input_data = Tensor(np.random.randn(32, 3, 32, 32).astype(np.float32))  # 模拟32张32x32 RGB图片
labels = Tensor(np.random.randint(0,10, (32)).astype(np.int32))

# 初始化组件
net = AscendNet()
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')  # 交叉熵损失[3](@ref)

# 构建训练单元
net_with_loss = nn.WithLossCell(net, loss_fn)

# 执行训练迭代一次，执行序比较一般需要一个step的数据即可
output = net_with_loss(input_data, labels)
print(f"Loss: {loss.asnumpy()}")
```

## 步骤二：执行标杆代码生成执行需文件

```python
import os

import mindspore as ms
from mindspore import nn, ops, Tensor
import numpy as np

# 设置执行序环境变量
os.environ["MS_ALLOC_CONF"] = "memory_tracker:True" 
# 设置昇腾设备环境
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")  # 强制启用图模式加速[1,5]
ms.set_context(device_id=0)  # 指定第0号昇腾芯片

class AscendNet(nn.Cell):
    def __init__(self):
        super().__init__()
        # 定义5个算子
        self.conv = nn.Conv2d(3, 64, kernel_size=3, pad_mode='same')  # 卷积算子
        self.gelu = ops.GeLU()                                      # 激活函数
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)           # 池化算子
        self.flatten = nn.Flatten()                                 # 展平算子
        self.fc = nn.Dense(64 * 16 * 16, 10)                            # 全连接层

    def construct(self, x):
        x = self.conv(x)        # 3通道→64通道特征提取
        x = self.gelu(x)        # 非线性激活
        x = self.pool(x)        # 下采样降维
        x = self.flatten(x)     # 数据展平
        x = self.fc(x)          # 全连接分类
        return x

# 生成模拟数据（替代真实数据集）
input_data = Tensor(np.random.randn(32, 3, 32, 32).astype(np.float32))  # 模拟32张32x32 RGB图片
labels = Tensor(np.random.randint(0,10, (32)).astype(np.int32))

# 初始化组件
net = AscendNet()
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')  # 交叉熵损失[3]

# 构建训练单元
net_with_loss = nn.WithLossCell(net, loss_fn)

# 执行训练迭代一次，执行序比较一般需要一个step的数据即可
output = net_with_loss(input_data, labels)
print(f"Loss: {loss.asnumpy()}")
```

生成的执行序文件可参见文件夹中的tracker_graph_relu.txt和tracker_graph_gelu.txt。


## 步骤三：将生成的执行序进行比较
### 1. 设置执行序文件
### 2. 选择比较区域
网络较小，之间选择整图呈现即可。
### 3. 点击比较
点击后可立即看到比较结果。结果中直接呈现出执行序的差异。