# migrator.api_dump

## troubleshooter.migrator.api_dump_init

> troubleshooter.migrator.api_dump_init(net, output_path='./ts_api_dump', *, retain_backward=False)

### 参数

- net(`torch.nn.Module`/`mindspore.nn.Cell`): 需要保存数据的网络，支持 MindSpore 的 Cell 和 Torch 的 Module。
- output_path(`str`):输出文件保存的路径。
- retain_backward(`bool`): 是否保存反向梯度数据，默认值为 `False`，不保存反向数据。

> 注意：
>
> 初始化时，工具会将`dropout`/`Dropout`相关API的`p`设置为0，即输出数据与输入数据相同，以固定不同框架的随机性。

### 数据格式

MindSpore 生成的目录结构。

```
output_path # 输出目录
└── rank0
    ├── mindspore_api_dump # npy数据目录
    ├── mindspore_api_dump_info.pkl # dump的info信息
    └── mindspore_api_dump_stack.json # dump的堆栈信息
```

Torch 生成的目录结构。

```
output_path # 输出目录
└── rank0
    ├── torch_api_dump # npy数据目录
    ├── torch_api_dump_info.pkl # dump的info信息
    └── torch_api_dump_stack.json # dump的堆栈信息
```

- dump的数据为**正向输入与输出**，**反向梯度的输入**，每个数据单独保存到一个npy文件，数据名称格式如下：

  ```
  [NN,Tensor,Functional]_NAME_cnt_[forward,backward]_[input, output].[idx]
  ```

  其中[NN,Tensor,Functional]表示接口的类型；NAME表示API的名称；cnt表示API调用的顺序；[forward,backward]表示正/反向数据；[input, output]表示数据的输入，输出；idx表示多输入或输出时数据的编号。

  例如"Functional_pad_0_forward_input.0.npy" 表示函数接口`pad`在正向第0次执行时输入的第0号位置元素的数据；

  "Tensor_permute_3_backward_input" 表示`Tensor`接口`permute`在反向第3次执行时的输入的梯度数据。

- `api_dump_info.pkl`文件为网络在dump时按照API的执行顺序保存的信息，文件项格式如下：
  ```
  [数据名称，保留字段，保留字段，数据类型，数据shape，[最大值，最小值，均值]]
  ```
  当数据为bool类型或关闭统计信息保存时，最大值/最小值/均值会显示为`NAN`。

- `api_dump_stack.json`为网络dump时按照API的执行顺序保存的堆栈信息，在精度出现问题时，可以根据相应的文件名找到对应的代码行。由于网络反向堆栈信息过于单一，因此只保存了正向的堆栈信息。堆栈信息以API为单位，与npy数据名称不同，它不包含 [input, output].[idx] 字段。

## troubleshooter.migrator.api_dump_start

> troubleshooter.migrator.api_dump_start(mode='all', scope=None, dump_type='all', filter_data=True, filter_stack=True)

开启数据 dump。

### 参数

- mode(str, 可选)：dump 模式，目前支持 `'all'`、`'list'`、`'range'`、`'api_list'`，默认值 `'all'`。`'all'` 模式会 dump 全部 API 的数据；`'list'`、`'range'`、`'api_list'` 模式通过配合 `scope` 参数可以实现 dump 特定 API、范围、名称等功能。

- scope(list, 可选)：dump 范围。根据 `mode` 配置的模式选择 dump 的 API 范围。API 范围中的名称可以通过输出目录下的 `api_dump_info.pkl` 文件获取）。

  - `mode` 为 `'list'` 时，`scope` 为 dump 特定的文件列表，例如 `['Functional_softmax_1', 'NN_Dense_1', 'Tensor___matmul___1']`，只会 dump 列表中的三个 API；
  - `mode` 为 `'range'` 时，`scope` 为 dump 的区间范围，例如 `['NN_Dense_1', 'Tensor___matmul___1']`，会 dump 从`'NN_Dense_1'`直到`'Tensor__matmul___1'`的所有 API；
  - `mode` 为 `'api_list'` 时，`scope` 为 dump 特定的 API 列表，例如 `['relu', 'softmax', 'layernorm']`，会 dump 名称中含有 `relu`、`softmax`、`layernorm` 关键字的所有 API，不区分 `Tensor`、`Functional` 等方法类型。

- dump_type(`str`, 可选)：dump 保存的数据类型，目前支持 `'all'`、`'statistics'`、`'npy'`、`'stack'`， 默认值为 `'all'`。以下模式均会保存数据的堆栈信息（`api_dump_stack.json`）与执行顺序（`api_dump_info.pkl`）。

  - 为 `'all'`时会保存数据的统计信息（`api_dump_info.pkl`文件中数据的最大/最小/均值信息）和 npy 文件，速度最慢，存储空间占用大；
  - 为 `'npy'`时，**不会保存数据的统计信息**，会保存数据的npy文件，速度较`'all'`模式快，存储空间占用大；
  - 为 `'statistics'` 时，**不会保存npy文件**，会保存数据的统计信息，存储空间占用小，结合`api_dump_compare`，可以根据统计信息初步定位精度问题；
  - 为 `'stack'`时，只会保存数据的堆栈信息（`api_dump_stack.json`）与执行顺序（`api_dump_info.pkl`），运行速度最快，存储空间占用最小，常用于快速验证`api_dump_compare`中API映射结果。

- filter_data(`bool`, 可选)：是否开启 dump 数据过滤，默认值为 `True`。为 `True` 时，非浮点类型的 Tensor 和标量将会被过滤，不会被保存。

- filter_stack(`bool`, 可选)：是否开启堆栈信息过滤，默认值为 `True`。为 `True`时，会过滤掉 `MindSpore`/`Pytorch`以及`Troubleshooter`dump功能的堆栈信息，只保留用户代码。

## troubleshooter.migrator.api_dump_stop

> troubleshooter.migrator.api_dump_stop()

停止数据 dump。

可以通过与`api_dump_start`配合使用，在网络执行中调用，对特定的范围的数据进行dump。

## troubleshooter.migrator.api_dump_compare

> troubleshooter.migrator.api_dump_compare(origin_path, target_path, output_path, *, rtol=1e-4, atol=1e-4, equal_nan=False, ignore_unmatched=False)

数据 dump 比对，支持 MindSpore 和 Torch 对比以及 MindSpore 同框架对比。

会根据目录下的文件名前缀自动识别是 MindSpore 还是 Torch 保存的数据，自动选择对比策略。

### 参数

- origin_path(`str`)：原始目录，与 init 接口的 output_path 同级。
- target_path(`str`)：目标目录，与 init 接口的 output_path 同级。
- output_path(`str`, 可选)：输出数据目录，默认值为None，不输出到文件。不为None时，输出目录下会保存 `ts_api_mapping.csv`（API映射文件）、 `ts_api_forward_compare.csv`（正向比对结果）、`ts_api_backward_compare.csv`（反向比对结果）。
如果dump时保存了多个step的数据，并且在`api_dump_start`时`retain_backward`为`True`，`api_dump_compare`会根据正反向信息自动识别出step，并对每个step的比较结果添加编号保存（此编号是按照顺序添加的，与网络实际step编号可能不同）。

  ```
  output_path # 输出目录
  ├── ts_api_mapping_{step}.csv # api映射表
  ├── ts_api_forward_compare_{step}.csv # 正向比对信息
  └── ts_api_backward_compare_{step}.csv # 反向比对信息
  ```

- rtol(`float`, 可选): 相对误差，默认值为 `1e-4`，内部调用 `numpy.allclose`的参数。
- atol(`float`, 可选): 绝对误差，默认值为 `1e-4`，内部调用 `numpy.allclose`的参数。
- equal_nan(`bool`, 可选): 是否将nan视为相等，默认值为 `False`，内部调用 `numpy.allclose`的参数。
- ignore_unmatched(`bool`, 可选): 是否忽略未匹配项，默认值 `False`。当原始目录和目标目录下API调用不一致时，可能会导致部分API未匹配，对于未匹配项，会显示为None。为`True`时，未匹配项不会显示。
