# TroubleShooter Release Notes

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
