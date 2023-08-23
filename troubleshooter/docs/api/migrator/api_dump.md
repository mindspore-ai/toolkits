# migrator.api_dump

## troubleshooter.migrator.api_dump_init

> troubleshooter.migrator.api_dump_init(net, output_path='./ts_api_dump', *, retain_backward=False)

### 参数

- net(torch.nn.Module/mindspore.nn.Cell): 需要保存数据的网络，支持MindSpore的Cell和Torch的Module。
- output_path(str): 输出文件保存的路径，默认值为'./ts_api_dump'，即保存在当前路径下的'ts_api_dump'目录。
- retain_backward(bool): 是否保存反向梯度数据，默认值为`False`，不保存反向数据。

> 警告：
>
> API级别数据保存依赖框架的hook机制，由于PyTorch对原地修改的API注册反向hook会抛异常（详细信息请参考[register_full_backward_hook](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook)）。**因此在保存反向梯度数据时，需要将网络脚本中原地修改的操作替换为非原地修改**（例如将`relu_`替换为`relu`，`torch.nn.ReLU`的`inplace`参数设置为`False`等），并在`api_init_dump`中设置`retain_backward`参数为`True`。

MindSpore生成的目录结构。

```
output_path # 输出目录
├── rank0
│   ├── mindspore_api_dump # npy数据目录
│   ├── mindspore_api_dump_info.pkl # dump的info信息
│   └── mindspore_api_dump_stack.json # dump的堆栈信息
└── rank1 # 多卡，暂不支持
```

Torch 生成的目录结构。

```
output_path # 输出目录
├── rank0
│   ├── torch_api_dump # npy数据目录
│   ├── torch_api_dump_info.pkl # dump的info信息
│   └── torch_api_dump_stack.json # dump的堆栈信息
└── rank1 # 多卡，暂不支持
```

## troubleshooter.migrator.api_dump_start

> troubleshooter.migrator.api_dump_start(mode='all', scope=[], dump_type='all', filter_data=True)

开启数据 dump。

### 参数

- mode(str)：dump模式，可选`'all'`、`'list'`、`'range'`、`'api_list'`，默认值`'all'`。`'all'`模式会dump全部API的数据；`'list'`、`'range'`、`'api_list'`模式通过配合`scope`参数可以实现dump特定API、范围、名称等功能。
- scope(list)：dump范围。根据`mode`配置的模式选择dump的API范围。API范围中的名称可以通过输出目录下的`api_dump_info.pkl`文件获取）。
  - `mode`为`'list'`时，`scope`为dump特定的文件列表，例如`['Functional_softmax_1_forward', 'NN_Dense_1_backward', 'Tensor___matmul___1_forward']`，只会dump列表中的三个API；
  - `mode`为`'range'`时，`scope`为dump的区间范围，例如`['NN_Dense_1_backward', 'Tensor___matmul___1_forward']`，会dump从'NN_Dense_1_backward'直到'Tensor___matmul___1_forward'的所有API；
  - `mode`为`'api_list'`时，`scope`为dump特定的API列表，例如`['relu', 'softmax', 'layernorm']`，会dump名称中含有relu、softmax、layernorm关键字的所有API，不区分`Tensor`、`Functional`等方法类型。
- dump_type(str)：dump保存的数据类型，可选`'all'`、`'statistics'`，默认值为`'all'`。为`'all'`时会保存数据的统计信息和npy文件；为`'statistics'`时，只会保存数据的统计信息。
- filter_data(bool)：是否开启dump数据过滤，默认值为`True`。为`True`时，非浮点类型的Tensor和标量将会被过滤，不会被保存。

## troubleshooter.migrator.api_dump_stop

> troubleshooter.migrator.api_dump_stop()

停止数据 dump。

## troubleshooter.migrator.api_dump_compare

> troubleshooter.migrator.api_dump_compare(origin_path, target_path, output_path)

数据 dump 比对，支持 MindSpore 和 Torch 对比以及 MindSpore 同框架对比。

会根据目录下的文件名前缀自动识别是 MindSpore 还是 Torch 保存的数据，自动选择对比策略。

### 参数

- origin_path(str)：原始目录，与 init 接口的 output_path 同级。
- target_path(str)：目标目录，与 init 接口的 output_path 同级。
- output_path(str，可选)：输出数据目录，默认值为None，不输出到文件。不为None时，输出目录下会保存 `ts_api_mapping.csv`（API映射文件）、 `ts_api_forward_compare.csv`（正向比对结果）、`ts_api_backward_compare.csv`（反向比对结果）。
