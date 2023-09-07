# API级别网络结果自动比较

## 应用场景

PyTorch 网络迁移到 MindSpore，以及 MindSpore 不同后端/版本迁移时，容易产生精度问题。正向问题往往需要二分保存中间变量，反向问题定位更是困难。API级别自动化比较功能实现了PyTorch 和 MindSpore 网络正、反向 API 级别(MindSpore:nn/ops/tensor, PyTorch:nn/funcitonal/torch.xx/tensor)输入、输出数据的自动保存以及对比，可以极大提高迁移效率。

> 注意：
>
> 目前api_dump相关功能只支持MindSpore的PYNATIVE模式。

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

    > 注意
    >
    > 当前没有针对优化器等接口进行特定处理，dump时接口的内部操作会被保存，由于PyTorch与MindSpore内部实现逻辑差异较大，导致对比时API映射困难。因此dump时需要跳过优化器，在反向执行后优化器执行前停止数据dump。

以下分别为PyTorch、MindSpore固定随机行、数据保存的完整代码示例。

**PyTorch 网络 dump**

```python
import troubleshooter as ts

# 1. 在网络，数据集处理前固定随机数
ts.fix_random()

# 2. 创建训练网络
net = create_net()
loss = torch.nn.CrossEntropyLoss()
pg = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
net.train()
optimizer.zero_grad()

# 3. 统一数据样本
image = torch.tensor(np.load('image.npy'))
label = torch.tensor(np.load('label.npy'))

# 4. 保存权重和转换映射，用于在MindSpore加载
ts.migrator.save_net_and_weight_params(net, path='pt_net_info')

# 5. 设置dump的网络
ts.migrator.api_dump_init(net, output_path="torch_dump", retain_backward=True)

# 6. 在迭代开始时开启dump
ts.migrator.api_dump_start()

# 7. 执行训练流程
pred = net(images)
loss = loss_function(pred, labels)
loss.backward()

# 8. 在反向计算结束，优化器更新前关闭dump
ts.migrator.api_dump_stop()

# 9. 执行优化器更新
optimizer.step()
```

**MindSpore 网络 dump**

```python
import troubleshooter as ts

# 1. 在网络，数据集处理前固定随机数
ts.fix_random()

# 2. 创建训练网络
net = create_net()
loss = mindspore.nn.CrossEntropyLoss()
optimizer = ms.nn.SGD(net.trainable_params(), learning_rate=args.lr, momentum=0.9, weight_decay=5E-5)
def forward_fn(data, label):
    logits = net(data)
    loss = loss_function(logits, label)
    return loss, logits
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

# 3. 统一数据样本
image = ms.Tensor(np.load('image.npy'))
label = ms.Tensor(np.load('label.npy'))

# 4. 保存权重和转换映射，用于在MindSpore加载
ts.migrator.convert_weight_and_load(weight_map_path="pt_net_info/torch_net_map.json",
                                    pt_file_path="pt_net_info/torch_troubleshooter_create.pth",
                                    net=net)

# 5. 设置dump的网络
ts.migrator.api_dump_init(net, output_path="ms_dump", retain_backward=True)

# 6. 在迭代开始时开启dump
ts.migrator.api_dump_start()

# 7. 执行训练流程
(loss, pred), grads = grad_fn(image, label)

# 8. 在反向计算结束，优化器更新前关闭dump
ts.migrator.api_dump_stop()

# 9. 执行优化器更新
optimizer(grads)
```

在执行完dump后，会在指定的目录下生成堆栈信息、执行信息、`npy`文件等数据，详细的数据格式说明请参考[数据格式](./api/migrator/api_dump.md#数据格式)。

### 数据对比

比较时是需要输入dump路径，即可自动识别是MindSpore还是PyTorch网络dump的数据，实现PyTorch或MindSpore迁移的对比。`api_dump_compare`还可以设置比较的精度、未匹配项的显示方式等，具体请参考[api_dump_compare](./api/migrator/api_dump.md#troubleshootermigratorapi_dump_compare)。

```python
import troubleshooter as ts

# 根据dump得到的数据进行比对，指定对比结果输出到output_path目录下
ts.migrator.api_dump_compare('torch_dump', 'ms_dump', output_path='compare_result')
```

对比结果一共分为三个部分，第一部分是 API 映射关系，第二部分是正向对比结果，如果对反向进行了 dump，还会有第三部分反向的对比结果。会生成 ts_api_mapping.csv（API 映射文件）、 ts_api_forward_compare.csv（正向比对结果）、ts_api_backward_compare.csv（反向比对结果）。

**API 映射关系**

| ORIGIN NET (pytorch) | TARGET NET (mindspore) |
| -------------------- | ---------------------- |
| NN_Conv2d_0          | NN_Conv2d_0            |
| Tensor_flatten_0     | Tensor_flatten_0       |
| Tensor_transpose_0   | Tensor_swapaxes_0      |
| NN_Identity_0        | NN_Identity_0          |

**正向结果对比**

The forward comparison results

正向对比结果按照执行的顺序进行比较，差异项分别为`numpy.allclose`、`allclose`达标比例、余弦相似度、差异值的 $mean$ / $max$ 统计量等信息。

| ORIGIN NET (pytorch)                   | TARGET NET (mindspore)                | shape of orig    | shape of target  | result of allclose | ratio of allclose | cosine similarity | mean & max diffs |
| -------------------------------------- | ------------------------------------- | ---------------- | ---------------- | ------------------ | ----------------- | ----------------- | ---------------- |
| NN_Conv2d_0_forward_input.0.npy        | NN_Conv2d_0_forward_input.0.npy       | (8, 3, 224, 224) | (8, 3, 224, 224) | True               | 100.00%           | 1.00000           | 0.00000, 0.00000 |
| NN_Conv2d_0_forward_output.npy         | NN_Conv2d_0_forward_output.npy        | (8, 768, 14, 14) | (8, 768, 14, 14) | True               | 100.00%           | 1.00000           | 0.00000, 0.00000 |
| Tensor_flatten_0_forward_input.0.npy   | Tensor_flatten_0_forward_input.0.npy  | (8, 768, 14, 14) | (8, 768, 14, 14) | True               | 100.00%           | 1.00000           | 0.00000, 0.00000 |
| Tensor_flatten_0_forward_output.npy    | Tensor_flatten_0_forward_output.npy   | (8, 768, 196)    | (8, 768, 196)    | True               | 100.00%           | 1.00000           | 0.00000, 0.00000 |
| Tensor_transpose_0_forward_input.0.npy | Tensor_swapaxes_0_forward_input.0.npy | (8, 768, 196)    | (8, 768, 196)    | True               | 100.00%           | 1.00000           | 0.00000, 0.00000 |
| Tensor_transpose_0_forward_output.npy  | Tensor_swapaxes_0_forward_output.npy  | (8, 196, 768)    | (8, 196, 768)    | True               | 100.00%           | 1.00000           | 0.00000, 0.00000 |

**反向对比结果**

The backward comparison results

反向结果显示为正向的逆序

| ORIGIN NET (pytorch)                 | TARGET NET (mindspore)                 | shape of orig | shape of target | result of allclose | ratio of allclose | cosine similarity | mean & max diffs |
| ------------------------------------ | -------------------------------------- | ------------- | --------------- | ------------------ | ----------------- | ----------------- | ---------------- |
| NN_Linear_48_backward_input.npy      | NN_Dense_48_backward_input.0.npy       | (8, 5)        | (8, 5)          | True               | 100.00%           | 0.99999           | 0.00000, 0.00000 |
| NN_Identity_1_backward_input.npy     | NN_Identity_1_backward_input.0.npy     | (8, 768)      | (8, 768)        | True               | 100.00%           | 1.00000           | 0.00000, 0.00000 |
| NN_LayerNorm_24_backward_input.npy   | NN_LayerNorm_24_backward_input.0.npy   | (8, 197, 768) | (8, 197, 768)   | True               | 100.00%           | 1.00000           | 0.00000, 0.00000 |
| Tensor___add___24_backward_input.npy | Tensor___add___24_backward_input.0.npy | (8, 197, 768) | (8, 197, 768)   | True               | 100.00%           | 1.00000           | 0.00000, 0.00000 |
| NN_Linear_47_backward_input.npy      | NN_Dense_47_backward_input.0.npy       | (8, 197, 768) | (8, 197, 768)   | True               | 100.00%           | 1.00000           | 0.00000, 0.00000 |

可以根据正反向对比结果，找到开始出现差异的api进行定位，减少二分的时间成本，提高定位效率。
