# TroubleShooter API

## 权重转换

权重转换教程，请参考[pth到ckpt权重自动转换](migrator.md#应用场景1pth到ckpt权重自动转换)

| 接口名                                                       | 描述                                                |
| ------------------------------------------------------------ | --------------------------------------------------- |
| [troubleshooter.migrator.get_weight_map](api/migrator/get_weight_map.md) | 生成PyTorch模型转换到MindSpore的参数映射文件        |
| [troubleshooter.migrator.convert_weight](api/migrator/convert_weight.md) | 转换PyTorch的模型参数(pth)为MindSpore模型参数(ckpt) |
| [troubleshooter.migrator.convert_weight_and_load](api/migrator/convert_weight_and_load.md) | 转换PyTorch的离线模型并加载到MindSpore模型中        |

##  模型权重参数比较

模型权重参数比较教程，请参考[pth与ckpt比较](migrator.md#应用场景2比对mindspore与pytorch的ckptpth)

| 接口名                                                       | 描述                           |
| ------------------------------------------------------------ | ------------------------------ |
| [troubleshooter.migrator.compare_ms_ckpt](api/migrator/compare_ms_ckpt.md) | 比较两个MindSpore的模型参数    |
| [troubleshooter.migrator.compare_pth_and_ckpt](api/migrator/compare_pth_and_ckpt.md) | 比较PyTorch和MindSpore模型参数 |

## 数据保存

| 接口名                             | 描述                                   |
| ---------------------------------- | -------------------------------------- |
| [troubleshooter.save](api/save.md) | 用于保存MindSpore和PyTorch的Tensor数据 |
|[troubleshooter.migrator.save_net_and_weight_params](api/migrator/save_net_and_weight_params.md)|将网络对象保存成文件|

## 数据比较

| 接口名                                                       | 描述                    |
| ------------------------------------------------------------ | ----------------------- |
| [troubleshooter.migrator.compare_npy_dir](api/migrator/compare_npy_dir.md) | 比较两个目录下的npy文件 |
| [troubleshooter.migrator.compare_list_dir](api/migrator/compare_list_dir.md) | 比较两个目录使用[save](api/save.md)保存的列表类型Tensor |
| [troubleshooter.migrator.compare_dict_dir](api/migrator/compare_dict_dir.md) | 比较两个目录使用[save](api/save.md)保存的字典类型Tensor |
| [troubleshooter.migrator.compare_grads_dir](api/migrator/compare_grads_dir.md) | 比较梯度                |

以下接口用于获取**name_map_list**

| 接口名                                                       | 描述                                |
| ------------------------------------------------------------ | ----------------------------------- |
| [troubleshooter.migrator.get_name_map_list_by_name](api/migrator/get_name_map_list.md) | 根据name完全相同的规则获取匹配列表  |
| [troubleshooter.migrator.get_name_map_list_by_number](api/migrator/get_name_map_list.md) | 根据number顺序获取匹配列表          |
| [troubleshooter.migrator.get_name_map_list_by_shape_edit_distance](api/migrator/get_name_map_list.md) | 根据shape的最小编辑距离获取匹配列表 |

## 自动化比较

| 接口名                                                       | 描述                    |
| ------------------------------------------------------------ | ----------------------- |
| [troubleshooter.migrator.NetDifferenceFinder](api/migrator/NetDifferenceFinder.md) | 自动化比较网络输出结果 |

## 工具插件

| 接口名                                                       | 描述                    |
| ------------------------------------------------------------ | ----------------------- |
| [troubleshooter.widget.fix_random](api/widget/fix_random.md) | 固定随机性 |
| [troubleshooter.toolbox.precision_tracker](api/toolbox/precision_tracker.md) | 算子升/降精度标识工具 |
