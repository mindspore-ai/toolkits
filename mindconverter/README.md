# Mindconverter 文档

> 当前版本：BETA

View English

##  介绍

MindConverter是一款用于将PyTorch脚本转换到MindSpore脚本的工具。结合转换报告的信息，用户只需对转换后的脚本进行微小的改动，即可快速将PyTorch框架的模型迁移到MindSpore。

##  安装

简易安装：

运行 `python setup.py install`即可完成安装

## 使用方法（命令行使用）

```
mindconverter [-h] [--version] --in_file IN_FILE [--output OUTPUT]
              [--report REPORT]

可选参数列表:
  -h, --help         显示此帮助消息并退出
  --version          显示程序版本号并退出
  --in_file IN_FILE  指定源代码文件的路径
  --output OUTPUT    指定转换后的代码的存储路径，默认情况输出位于当前的工作目录
  --report REPORT    指定转换日志的存储路径，默认情况输出位于当前的工作目录
```

请注意：由于mindspore和pytorch架构存在的一些差异，本工具并不能完美进行转换，相关的信息以输出报告的形式存储下载，请您在转换完毕后比对输出日志进行调整

### 不支持转换的类型

- `Dtype`参数，由于mindspore和pytorch的数据类型不同，该参数无法被转换，需要您根据代码进行调整；
- 参数类型为`Tuple`,`Dict`的算子，由于某些算子，mindspore暂不支持以元组形式传参，故此暂不支持对该类型的算子进行转换，在输出报告中会给出相应的解决方案和对应的文档链接，需要您进行比对然后手动进行修改；

## 使用样例

下面主要是对于两类使用场景下进行样例解析：

1. 对于简单模型「Mindconverter 完全支持的算子类型」

  对于Pytorch实现的`lenet.py`模型如下：

```python
import torch.nn as nn
import torch.nn.functional as F


class TestLeNet(nn.Module):
    """TestLeNet network."""
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input_x):
        """Callback method."""
        out = F.relu(self.conv1(input_x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
```

  - 安装Mindconverter后，使用命令行指令 `mindconverter --in_file lenet.py`，如果未制定报告输出目录`–output`及模型输出目录`–report`，将在当前生成`output` 文件夹，存放转换后的模型代码，同时在当前目录下生成`MODEL_report`报告文件，报告文件格式：

```
[开始转换]
[插入] 'import mindspore.ops' 已经插入到被转换为文件中
[插入] 'import mindspore.experimental.optim as optim' 已经插入到被转换为文件中
行 1:列 0: [已转换] 'import torch.nn as nn' 已经被转换为了 'import mindspore.nn as nn'.
行 2:列 0: [已转换] 'import torch.nn.functional as F' 已经被转换为了 'import mindspore.ops as ops'.
行 5:列 16: [已转换] 'nn.Module' 已经被转换为了 'nn.Cell'.
行 8:列 21: [已转换] 'nn.Conv2d' 已经被转换为了 'nn.Conv2d'. 参数已转换，本算子MindSpore与Pytorch存在一些差异，参考资料： MindSpore：与PyTorch实现的功能基本一致，但存在偏置差异和填充差异。 https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_diff/Conv2d.html
行 9:列 21: [已转换] 'nn.Conv2d' 已经被转换为了 'nn.Conv2d'. 参数已转换，本算子MindSpore与Pytorch存在一些差异，参考资料： MindSpore：与PyTorch实现的功能基本一致，但存在偏置差异和填充差异。 https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_diff/Conv2d.html
行 10:列 19: [已转换] 'nn.Linear' 已经被转换为了 'nn.Dense'. MindSpore：MindSpore此API实现功能与PyTorch基本一致，而且可以在全连接层后添加激活函数。 https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_diff/Dense.html
行 11:列 19: [已转换] 'nn.Linear' 已经被转换为了 'nn.Dense'. MindSpore：MindSpore此API实现功能与PyTorch基本一致，而且可以在全连接层后添加激活函数。 https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_diff/Dense.html
行 12:列 19: [已转换] 'nn.Linear' 已经被转换为了 'nn.Dense'. MindSpore：MindSpore此API实现功能与PyTorch基本一致，而且可以在全连接层后添加激活函数。 https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_diff/Dense.html
行 14:列 4: [已转换] 'forward' 已经被转换为了 'construct'.
行 16:列 14: [已转换] 'F.relu' 已经被转换为了 'ops.relu'. 
行 17:列 14: [已转换] 'F.max_pool2d' 已经被转换为了 'ops.MaxPool'. 
行 18:列 14: [已转换] 'F.relu' 已经被转换为了 'ops.relu'. 
行 19:列 14: [已转换] 'F.max_pool2d' 已经被转换为了 'ops.MaxPool'. 
行 20:列 23: [已转换] 'out.size' 已经被转换为了 'out.shape'. 
行 21:列 14: [已转换] 'F.relu' 已经被转换为了 'ops.relu'. 
行 22:列 14: [已转换] 'F.relu' 已经被转换为了 'ops.relu'. 
[转换完毕]
```

报告文件将给出识别到的每个pytorch算子的转换方式，由于Mindspore和Pytorch框架存在诸多差异，建议用户参考报告内容，对转换后的模型代码再进行适当的微调。

2. 对于不支持模型的转换，需要根据报告文件进行微调

例如在`covLSTM.py`模型中，mindconverter对于`torch.zeros`的元组传入参数不支持，

```python
import torch.nn as nn
import torch

...
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        ...

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1) 

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

......
```

在输出报告中对应代码位置将显示

```
行 48:列 12: [已转换] 'torch.tanh' 已经被转换为了 'ops.tanh'. 
行 51:列 21: [已转换] 'torch.tanh' 已经被转换为了 'ops.tanh'. 
行 57:列 16: [未转换] 'torch.zeros' 没有进行转换，建议您自行进行参数转换 
行 58:列 16: [未转换] 'torch.zeros' 没有进行转换，建议您自行进行参数转换 
行 61:列 15: [已转换] 'nn.Module' 已经被转换为了 'nn.Cell'.
```

而在输出代码结果中当前代码并没有被转换并且会进行报错

```python
import mindspore.nn as nn
import mindspore.experimental.optim as optim
import mindspore.ops
import mindspore.ops as ops


class ConvLSTMCell(nn.Cell):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        ...
    def construct(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = ops.cat(tensors=[input_tensor, h_cur], axis=1)  

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = ops.split(tensor=combined_conv, split_size_or_sections=self.hidden_dim, axis=1)
        i = ops.sigmoid(input=cc_i)
        f = ops.sigmoid(input=cc_f)
        o = ops.sigmoid(input=cc_o)
        g = ops.tanh(input=cc_g)

        c_next = f * c_cur + i * g
        h_next = o * ops.tanh(input=c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device), #未进行转换
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)) #未进行转换
        
......
```

需要用户手动进行微调，例如：

```python
torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)

ops.zeros((batch_size, self.hidden_dim, height, width), dtype=None)
```





