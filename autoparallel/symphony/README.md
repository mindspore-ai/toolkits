# Taylor训练自动并行策略搜索工具

## 1 介绍

（1）工具的作用简介：

大规模LLM的训练有多种并行范式，在不同的并行配置和优化特性的混搭下性能差别很大。本工具用于大模型训练并行配置搜索，根据给定的模型信息和硬件信息，提供建议的并行训练配置、优化特性的启用与否以及流水线负载均衡的配置。

（2）使用工具的总体流程

用户提供模型网络脚本和一定的测试资源（推荐至少为全量的1/4），经ND配置搜索剪枝算法后会生成约20个需要做profile的yaml配置文件，人工做完profile之后工具将会自动生成各个配置的推荐流水线和重计算配置以及相应的Cost估值，估值越小的配置性能最优。

（3）支持范围：

已支持模型：

- [x] deepseekv3
- [x] llama

已支持特性：

- [x] DP
- [x] TP
- [x] PP
- [x] EP
- [x] SP
- [x] OP
- [x] 重计算
- [x] VPP

计划支持特性

- [ ] CP
- [ ] PBS
- [ ] swap



## 2 使用指南

### 2.1  ND配置搜索

（1）工具使用场景：

ND配置搜索为自动并行策略搜索的第一阶段，搜索出用户给定模型和卡数下合法且预估内存不超限的DP, TP, PP, EP组合配置集合，并根据可用资源数量生成要做profile的yaml配置文件。

（2）使用方法：

在已安装MindSpore的环境中，执行脚本parallel_tool.py，并根据需求填写如下参数，则会自动执行搜索，并做约20~100次dryrun：

输入参数：

| 参数        | 含义                  | 默认值                          |
|-----------|---------------------|------------------------------|
| yaml_path | 模型配置文件              | ./config/pretrain_deepseek3_128p.yaml |
| mindformers_dir    | run_mindformer.py路径 | ./mindformers       |
| dryrun_data_dir    | 已有dryrun数据路径        | None                         |
| profile_data_dir   | 已有profile数据路径       | None                         |
| parallel_num       | dryrun并行数           | 2                            |
| rank_num           | 可用于做profile的卡数      | 64                           |
| solver_name        | 用于下一阶段流水线搜索的求解器名称   | HIGHS                       |
| register_path      | 模型注册路径              | ./register/                         |
| small_dataset  | 数据量较小的数据集路径         | ./config/dataset/wiki4096.mindrecord |
| dataset        | 数据集路径               | ./config/dataset/wiki103-4k.mindrecord |
| max_expert_parallel| 搜索范围的最大EP数          | 64                           |
| output_path        | 输出信息的路径             | ./output/nd_output/          |
| env_json  | 环境变量配置文件            | ./config/env_config.json     |
| gbs                | global batch size   | 1024                         |

输出信息：

| 名称                            | 含义                                              |
|:------------------------------|-------------------------------------------------|
| nd_candidate_config.csv       | 包含所有候选ND配置的csv文件, 每行包含dp, tp, pp, ep, peakMem信息 |
| DPx_TPx_PPx_EPx_pretrain.yaml | 生成在output/profile_yaml目录下的yaml文件，用于做profile     |

使用示例：

```
python parallel_tool.py --yaml_path jiutian.yaml --mindfomres_dir ~/mindformers/run_mindformer.py --dataset_dir ~/wiki_4k/fastchat4096.mindrecord --gbs 9216 --parallel_num 16 
```


### 2.2 ND搜索空间寻优
#### 2.2.1 使用场景
     
   ND搜索空间生成后，为进一步筛选优选配置，基于各项配置及其对应的流水线最优配置跑出端到端耗时，并根据端到端耗时对各项配置进行排序。

#### 2.2.2 前置条件
1、ND搜索完成，各个并行配置已写入对应YAML；

2、各个并行配置已做profile，从profile中解析到如下信息：

| 参数           | 含义                     |          默认值          | 必须 | 
|--------------|------------------------|:---------------------:|:--:|
| dmratio    | 单层dense计算时间/单层moe计算时间  |          0.3          | 是  |
| bfratio | 单层后向计算时间/单层前向计算时间      |           2           | 是  |
| re_grow_ration | 单层重计算计算时间/单层前向计算时间     |           1           | 是  |
| hratio | head loss计算时间/单层前向计算时间 |          1.5          | 是  |
| moe_bw | 单层moe计算实际时间            |         None          | 是  |

#### 2.2.3 使用方法
1、主程序python文件

    pipeline_tool.py
2、可配置参数

| 配置项                | 含义                    | 默认值                             | 必须 | 
|--------------------|-----------------------|---------------------------------|:--:|
| --yaml_path        | ND搜索生成的YAML文件夹路径      | ./output/dryrun_yaml/           | 是  |
| --mindformers_dir  | run_mindformer.py路径   | ./mindformers/run_mindformer.py | 是  |
| --parser_result  |  各yaml profile结果汇总文件路径   | None                            |  是  |
| --profile_data_dir | profile数据路径           | ./profile_data/                 | 否  |
| --solver_name      | 流水线使用的求解器名称           | HIGHS                           | 否  |
| --gbs              | global batch size     | 1024                            | 否  |
| --nd_path          | 包含所有候选ND配置的csv文件      | /config/nd_result.csv           | 否  |

3、使用示例

```
python pipeline_tool.py --yaml_path ./output/dryrun_yaml/ --mindfomres_dir ~/mindformers/run_mindformer.py --parser_result ./config/profiling_result.csv
```
4、输出信息

| 名称                                                | 含义                         | 必须 |
|---------------------------------------------------|----------------------------|----|
| 运行目录/pipeline_output/test_result_%Y%m%d%H%M%S.csv | topN的并行配置及对应的最优流水线配置的csv文件 | 是  |


### 2.3 流水线配置搜索
#### 2.3.1 使用场景


求解流水线并行配置

#### 2.3.2 前置条件

1、某一确定的ND并行配置；

2、可拉起dryrun的yaml文件，yaml文件中除offset、recompute_config无需配置正确外，其余须与实际配置保持一致

#### 2.3.3 使用方法
1、主程序python文件
    
    pipeline_parallel.py

2、可配置参数

| 参数           | 含义                     | 默认值    | 必须 |
|--------------|------------------------|--------|----|
| -yaml        | 需要求解的模型配置文件            | None   | 是  |
| -mindformers | run_mindformer.py路径    | None   | 是  |
| -solver_name | 流水线使用的求解器名称            | HIGHS  | 否  |
| -layer_ratio | dense前向计算时间/前向计算时间     | 0.33   | 是  |
| -b_ratio     | 反向计算时间/正向计算时间          | 2.0    | 是  |
| -head_loss   | head loss计算时间/前向计算时间   | 1.5    | 是  |
| -ra_ratio    | 重计算时间/前向计算时间           | 1      | 是  |
| -t           | 限制求解器的求解时间             | sys.maxsize | 否  |
| -dryrun      | 是否使用dryrun计算内存信息       | True   | 否  |
| -check       | 是否使用double_check进行内存拟合 | True   | 否  |
| -extract     | 是否提取单独solution信息       | False  | 否  |
| -solution    | 需要提取的solution文件路径      | None   | 否  |
注：以上参数必须与否按照是否影响最终结果评判

3、使用示例

```
python pipeline_parallel.py -yaml jiutian.yaml -mindformers ~/mindformers/run_mindformer.py -dryrun n -check n -layer_ratio 0.3 -b_ratio 2 -head_loss 1.5 -ra_ratio 1
```

4、输出信息

| 名称                         | 含义                                          | 必须 |
|----------------------------|---------------------------------------------|----|
| logger日志                   | 打印offset、recompute_config等信息                | 是  |
| yaml文件                     | 在配置的yaml文件中写入求解的最优offset、recompute_config信息 | 是  |
| 运行目录/pipeline_output/~.mps | 建模的数学规划文件                                   | 否  |
