# API级别逐层比较网络结果

## 应用场景

PyTorch 网络迁移到 MindSpore，以及 MindSpore 不同后端/版本迁移时，容易产生精度问题。正向问题往往需要二分保存中间变量，反向问题定位更是困难。API级别逐层比较功能实现了PyTorch 和 MindSpore 网络正、反向 API 级别(MindSpore:nn/ops/tensor, PyTorch:nn/funcitonal/torch.xx/tensor)输入、输出数据的自动保存以及对比，大大提高迁移效率。

在网络输出结果出现差异时，需要逐层对比网络的输出结果时，可以使用troubleshooter的api_dump相关接口进行数据保存与自动化对比，接口请参阅[api dump](./api/migrator/api_dump.md)。

## 如何使用

精度对比时，首先需要保证三个条件，即需要满足随机性固定并相同、输入数据样本一致、初始化权重一致。具体可参考[网络精度对比三个条件](https://mindspore.cn/docs/zh-CN/master/migration_guide/migrator_with_tools.html#%E7%BD%91%E7%BB%9C%E6%AF%94%E5%AF%B9%E4%B8%89%E4%B8%AA%E5%9F%BA%E6%9C%AC%E6%9D%A1%E4%BB%B6)，以下示例中只简要说明。

> 注意：
>
> 针对dropout类随机接口，api_dump会自动将p置为0，即输出数据和输入数据相同。

在满足三个条件后，就可以进行数据保存以及数据对比了。

### 数据保存
在PyTorch迁移场景下，需要分别对 MindSpore 和 PyTorch 网络进行 API 级整网数据 dump。

在MindSpore的不同版本/后端迁移场景下，需要对MindSpore网络进行API级整网数据dump。

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
ts.save_net_and_weight_params(net, path='pt_net_info')

# 5. 设置dump的网络
ts.migrator.api_dump_init(net, output_path="torch_dump", retain_backward=True)

# 6. 在迭代开始时开启dump
ts.migrator.api_dump_start()

# 7. 执行训练流程
pred = net(images)
loss = loss_function(pred, labels)
loss.backward()

# 8. 在迭代结束后关闭dump
ts.migrator.api_dump_stop()
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

# 8. 在迭代结束后关闭dump
ts.migrator.api_dump_stop()
```

#### 数据对比

比较时是需要输入dump路径，即可自动识别是MindSpore还是PyTorch网络dump的数据，实现PyTorch或MindSpore迁移的对比。

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

正向结果对比结果为顺序

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
