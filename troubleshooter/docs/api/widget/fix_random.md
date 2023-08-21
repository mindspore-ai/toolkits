## troubleshooter.fix_random

> troubleshooter.fix_random(seed=16)

固定python、numpy、pytorch、mindspore等随机性。

包括：

| API                                      | 固定随机数                    |
| ---------------------------------------- | ----------------------------- |
| os.environ['PYTHONHASHSEED'] = str(seed) | 禁止Python中的hash随机化      |
| random.seed(seed)                        | 设置random随机生成器的种子    |
| np.random.seed(seed)                     | 设置numpy中随机生成器的种子   |
| torch.manual_seed(seed)                  | 设置当前CPU的随机种子         |
| torch.cuda.manual_seed(seed)             | 设置当前GPU的随机种子         |
| torch.cuda.manual_seed_all(seed)         | 设置所有GPU的随机种子         |
| torch.backends.cudnn.enable=False        | 关闭cuDNN                     |
| torch.backends.cudnn.benchmark=False     | cuDNN确定性地选择算法         |
| torch.backends.cudnn.deterministic=True  | cuDNN仅使用确定性的卷积算法   |
| mindspore.set_seed(seed)                 | 设置mindspore随机生成器的种子 |

### 参数

- seed（`int`，可选）：随机数种子，默认值16。

### 用例

```python
# The following code runs repeatedly with the same result
import troubleshooter as ts
import numpy as np
import mindspore as ms
import torch
ts.fix_random()

np_a = np.random.rand(1, 3, 2)
ms_a = ms.ops.rand(1, 3, 2)
torch_a = torch.rand(1, 3, 2)
print(np_a)
print(ms_a)
print(torch_a)
```
