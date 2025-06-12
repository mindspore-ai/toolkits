# 配置文件


配置文件`toolkit\pipeline_balance\cfgs\auto_ppb_config.yaml` 列出了一些使用工具时需修改的常用配置，同时也有使用自定义化功能时需修改的配置，下面逐一介绍。


## profiling_config
```yaml
profiling_config:
  micro_batch_num: 8
  folder_path: "/your/path/here"

  head_layers: ["LlamaEmbedding"]
  body_layers: ["DeepSeekV2DecodeLayer"]
  tail_layers: ["lm_head-Linear"]

  body_layers_ends: ""
```
用于分析模型的时间信息的profiling数据，可以帮助工具更精确的分析模型的时间信息。 如果提供的profiling数据可用，工具会优先使用profiling数据。
| 参数 | 含义 | 取值范围 |
| :----: |----|:----:|
| `micro_batch_num` | 生成的profiling数据时配置的micro_batch_num | int|
| `folder_path` | 存放profiling文件的文件夹 | str|
| `head_layers` | head layers的name| list[str]|
| `body_layers` | body_layers的name| list[str]|
| `tail_layers` | tail_layers的name| list[str]|
| `body_layers_ends` | 异构body layers的结束位置如 DeepSeekV2DecodeLayer:[2, 61, 62] 表示三种body layer分别是0-2, 3-61, 62层| list[int]|

## time_config
```yaml
time_config:
  llama:
    head: 90
    body:
      LLamaDecodeLayer: [90]
    tail: 180

  deepseek:
    head: 18
    body:
      DeepSeekV2DecodeLayer: [17]
    tail: 25
```
模型的时间信息，`head` `body` `tail` 的数值表示，`llama`, `deepseek`对应配置文件中的`model_type`, 可自定义修改。如果需要工具直接读取，需完整提供`head` `body` `tail`的值。`body`的数据类型是`list[int]`, 对应从前置后的body layer的耗时。比如 `DeepSeekV2DecodeLayer: [7, 17, 20]` 表示dense, moe, mtp的耗时分别为7， 17， 20。


## backward_coef
```yaml
backward_coef:
  full: 0.5
  select: 0.04
  both: 0.165
  comm: 0.125
```

模型开启重计算占用的时间比例，可根据实际模型自定义需改， 参数含义如下


| 参数 | 含义 | 取值范围 |
| :----: |----|:----:|
| `full` |  全重计算| float, (0, 1)|
| `select` | 选择性重计算| float, (0, 1)|
| `both` | 选择性重计算叠加通信重计算 | float, (0, 1)|
| `comm` | 通信重计算| float, (0, 1)|


## 其他参数介绍
```yaml
validate: True

training_config_path: "/your/path/here"

model_type: "llama"
```
| 参数 | 含义 | 取值范围 |
| :----: |----|:----:|
| `validate` |  validate开关，开启后工具会自动验证算法给出的策略，对比实际每个stage的峰值显存占用和算法预计的峰值现存占用| bool|
| `training_config_path` | 用于dryrun和validate的training_config路径， 工具会修改其中的pipeline配置拉起dryrun| str|
| `model_type` | 对应`training_config`和`time_config`的模型类型| str|
