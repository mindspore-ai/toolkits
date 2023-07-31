## troubleshooter.widget.fix_random

> troubleshooter.widget.fix_random(seed)

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

- seed：随机数种子。
