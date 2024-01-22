## troubleshooter.widget.loss_compare
>
> troubleshooter.widget.loss_compare(left_file: dict, right_file: dict, title="Loss Compare")

### 参数

- left_file: 参数类型为字典，传入待比对文件的配置信息，具体配置请看示例。
- right_file: 参数类型为字典，传入待比对文件的配置信息，具体配置请看示例。
- title：图片的标题，默认为"Loss Compare"

### 样例

```python
import troubleshooter as ts


left_file = {
    # 比对文件日志路径
    "path": "npu_train_loss.txt",
    # 提取loss值的tag
    "loss_tag": "lm_loss:",
    # 图例label名称
    "label": "npu_loss"
}

right_file = {
    # 比对文件日志路径
    "path": "gpu_train_loss.txt",
    # 提取loss值的tag
    "loss_tag": "lm_loss:",
    # 图例label名称
    "label": "gpu_loss"
}

ts.widget.loss_compare(left_file, right_file)
```

### tag匹配规则
根据Tag的取值有如下3点匹配规则：

1. 匹配数据时将逐行读取文件的文本，查找是否存在传入的tag值，找到loss_tag则查找其后是否存在数字或以科学计数法表示的数字（忽略两者中间空格）。
2. 若存在多个匹配项，将第一项作为匹配值。


### 交付件
所有交付件保存在在当前目录下loss_compare文件夹下

1. loss比对的曲线图以及误差图。
2. 两份日志导出loss数据的csv文件。
3. 统计信息csv文件。

### 使用说明
1. 跟据loss_tag进行日志解析，生成的loss曲线比对图中横坐标为解析的loss值个数，并非日志中的迭代数。
2. 使用时两份日志的loss打印间隔请保持一直。
