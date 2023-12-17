# TroubleShooter Release Notes

## 1.0.16 Release Notes

### 主要特性和增强

#### API级别数据保存能力增强

- `save`接口能力增强，支持MindSpore `Ascend`后端并行场景，详见[save](docs/api/save.md)；
- 添加`save_grad`反向梯度保存接口，支持保存MindSpore与PyTorch中Tensor对应的反向梯度，详见[save_grad](docs/api/save_grad.md)；

#### API级别自动比较支持MindTorch

- 支持保存MindTorch网络API级别输入输出数据及模块(named_modules)名称，详见[api_dump文档](https://gitee.com/mindspore/toolkits/blob/master/troubleshooter/docs/api/migrator/api_dump.md)；
- 支持PyTorch与MindTorch网络数据对比，详见[API级别网络结果自动比较](https://gitee.com/mindspore/toolkits/blob/master/troubleshooter/docs/api_compare.md)；

### API变更

#### 非兼容性接口变更

- 接口名称：troubleshooter.save

  变更内容：删除auto_id参数，增加output_mode参数。

  说明：
    1. id 编号变化，原有方式可以指定是否编号，变更后每个数据都会有自增编号，不可去除；原有编号在保存嵌套类型时编号相同，变更后每个数据单独编号。
    2. 增加output_mode参数，默认为`'npy'`，与原有接口行为相同。另支持`'print'`，不保存文件，使用print直接输出。
    3. 新版本中，save使用MindSpore [TensorDump](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/ops/mindspore.ops.TensorDump.html)算子保存数据，当前暂不支持GPU和CPU的图模式。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> 1.0.16 接口 </td>
  </tr>
  <tr>
  <td><pre>
  save(file,
       data,
       auto_id=True,
       suffix=None)
  </pre>
  </td>
  <td><pre>
  save(file,
       data,
       suffix=None,
       output_mode='npy')
  </pre>
  </td>
  </tr>
</table>

### 问题修复

- 修复`NetDifferenceFinder`当网络和数据不在同一个device上运行失败的问题；

## 1.0.15 Release Notes

### 主要特性和增强

#### API级别自动比较

- 新增api_dump使用文档，[API级别网络结果自动比较](https://gitee.com/mindspore/toolkits/blob/master/troubleshooter/docs/api_compare.md)；
- api_dump_start支持按不同模式、范围、类型进行保存，支持过滤标量数据等功能，支持多个step数据保存，详见[api_dump_start文档](https://gitee.com/mindspore/toolkits/blob/master/troubleshooter/docs/api/migrator/api_dump.md#troubleshootermigratorapi_dump_start)；
- api_dump_compare支持自动识别dump的模式和框架，支持统计信息比较，多step比较，详见[api_dump_compare文档](https://gitee.com/mindspore/toolkits/blob/master/troubleshooter/docs/api/migrator/api_dump.md#troubleshootermigratorapi_dump_compare)；
- 支持dump torch原地修改API的反向dump；
- dropout类接口自动将p设置为0，移除随机性，方便精度对比；
- 优化器/print接口内部操作不dump，减少冗余信息干扰；

### 问题修复

- TroubleShooter移除对MinSpore/PyTorch包的依赖；
- 修复`PyTorch` api_dump时保存文件权限问题；
