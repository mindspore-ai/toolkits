# 测试文档

`unit.test.py`

### 算子模块的独立测试：
- 单独测试torch算子，``test_torch_migrate(torch_api, ms_template_api)``

- 单独测试torch.nn算子, `test_torch_nn_migrate(torch_nn_api, ms_nn_template_api)`

- 单独测试torch.Tensor算子，`test_tensor_migrate(tensor_api, ms_tensor_template_api)`

- 单独测试torch.optim算子，`test_torch_optim_migrate(torch_optim_api, ms_template_api)`

**测试样例：**

参数1：需要输入待测试的算子信息，例如：torch.abs(-123)

参数2：使用mindsporeAPI正确转换的算子信息，例如:ops.abs(-123)

**目标：**

测试程序会将torch.abs(-123)进行转换，并比较转换后的算子参数等信息是否与测试员给出的正确转换的算子信息一致

**示例：**

输入：torch.abs(-123), ops.abs(-123)

输入参数：test_torch_migrate(“torch.abs(-123)”, “ops.abs(-123)”)

=>Mindconverter[ torch.abs(-123) ] = ops.abs(-123) = ms_template_api

=>测试通过

### 完整模型文件测试：

**测试样例：**

`test_torch_mindspore_model(model_file_path, ms_template_file_path)`

参数1：待转换的torch模型代码

参数2：mindsporeAPI标准转换的模型代码

**目标：**

测试程序会将待转换的torch模型代码进行转换，然后与测试员给出的mindsporeAPI的标准转换代码进行比较，通过mindconverter转换后的代码与测试员的标准代码一致，则测试通过。
