## 代码结构

```
--\
--common\ 包含通用设置，日志和抛出异常
----__init__.py 初始化common
----exceptions.py 异常处理代码
----log.py 日志处理代码
--conf\ 配置文件
----__init__.py 初始化conf
----constants.py
----defaults.py
--mappings\ pytorch to mindspore映射代码
----f_mappings.json torch.functional映射
----nn_mappongs.json torch.nn映射
----tensor_dot_mappings.json torch.tensor映射
----torch_dot_mappings.json torch映射
----torch_optim_mappings.json torch.optim映射
--ops\
----f_list.json 包含支持的torch.functional的算子
----nn_list.json 包含支持的torch.nn的算子
----tensor_dot_list.jsonn 包含支持的torch.tensor的算子
----torch_dot_list.json 包含支持的torch的算子
----torch_optim_list.json 包含支持的torch.optim算子
--\
--__init__.py 初始化mindconvert
--_version.py 版本控制
--ast_edit.py AST语法树的节点编辑
--cli.py 程序处理入口
--code_analysis.py 代码结构分析
--config.py 完成对于mapping映射的相关控制和抛出警告
--converter.py 运行主程序
--forward_call.py 在脚本文件中找到forward函数调用的子函数。
--funcs.py 为某些算子生成显式映射表
--map_api_download.py 完成对于算子映射表的下载和参数的转换
--warn_infos\
----supported_warn_infos.json 支持算子但存在一些差异的算子的相关警告信息
----unsupported_warn_infos.json mindspore不支持的算子的输出警告信息
```



## 维护相关

1. 运行map_api_download.py程序将自动下载Mindspore和Pytorch最新版本的算子参数和名称，同时以Json格式存储到mappings文件夹和ops文件夹下

   - ops文件夹存储的是对于算子的名称
   - mappings文件夹存储的对应算子的参数以及pytorch和mindspore参数的映射

2. 注意下载完后，由于文档的差异性/文档格式不标准的问题，仍需手动进行调整，运行`map_api_download.py`时，存在问题的算子会在控制台进行输出，下面是已知存在问题的一下算子

   - ```
     torch.nn.CeLU
     mindspore.nn.Hsigmoid
     mindspore.nn.Hswish
     mindspore.nn.Identity
     mindspore.ops.uniform
     mindspore.nn.LogSigmoid
     mindspore.nn.LogSoftMax
     mindspore.nn.MultiLabelMarginLoss
     mindspore.nn.ReLU
     mindspore.nn.ReLU6
     mindspore.nn.SeLU
     torch.nn.SeLU
     mindspore.nn.Sigmoid
     mindspore.nn.SiLU
     mindspore.nn.Softmax2d
     mindspore.nn.Softsign
     mindspore.nn.Tanh
     mindspore.nn.Tanhshrink
     mindspore.ops.clip_by_value
     mindspore.ops.clip_by_norm
     torch.optim.AdaMax
     torch.optim.Optimizer
     torch.optim.RMSProp
     torch.optim.lr_scheduler.LRScheduler
     ```

3. 应该确保仓库的mappings文件夹和ops文件夹下的映射关系是最新的

4. `warn_infos`文件夹下存放算子的注意信息以及对应文档的链接

5. 如果转换过程中遇到问题，只需要找到出问题的算子（控制台会给出），然后在mappings文件夹下找到相应的算子，对算子的参数和映射关系进行调试修改即可