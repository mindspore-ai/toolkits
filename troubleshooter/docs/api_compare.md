# API级别逐层比较网络结果

在网络输出结果出现差异时，需要逐层对比网络的输出结果时，可以使用troubleshooter的api_dump相关接口进行数据保存与自动化对比，接口请参阅[api dump](./api/migrator/api_dump.md)。

## 数据dump

分别对MindSpore和PyTorch网络进行API级整网数据dump。

```python
import troubleshooter as ts

# 1. 在网络定义和数据集处理前固定随机数
ts.fix_random()

# 2. 对需要dump数据的网络进行初始化
ts.migrator.api_dump_init(net, output_path="out_path")

# 3. 在网络第一个迭代开始前开启dump
ts.migrator.api_dump_start()

# 4. 网络执行
net(xxx)

# 5. 在网络第一个迭代结束后关闭dump
ts.migrator.api_dump_stop()
```

## 数据对比

```python
import troubleshooter as ts

# origin_path与target_path为api_dump_init中的output_path
origin_path = "xx./ms_dump"
target_path = "xxx./torch_dump"
# 输出结果保存路径
output_path = "./compare_result"

# 对比完成之后会生成ts_api_mapping.csv（API映射文件）、 ts_api_forward_compare.csv（正向比对结果）、ts_api_backward_compare.csv（反向比对结果）
ts.migrator.api_dump_compare(origin_path, target_path, output_path)
```
