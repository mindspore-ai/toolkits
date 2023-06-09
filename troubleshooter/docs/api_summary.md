# TroubleShooter API

## 权重转换

| 接口名                                                       | 描述                                                |
| ------------------------------------------------------------ | --------------------------------------------------- |
| [troubleshooter.migrator.get_weight_map](api/migrator/get_weight_map.md) | 生成PyTorch模型转换到MindSpore的参数映射文件        |
| [troubleshooter.migrator.convert_weight](api/migrator/convert_weight.md) | 转换PyTorch的模型参数(pth)为MindSpore模型参数(ckpt) |
| [troubleshooter.migrator.convert_weight_and_load](api/migrator/convert_weight_and_load.md) | 转换PyTorch的离线模型并加载到MindSpore模型中        |

##  模型权重参数比较

| 接口名                                                       | 描述                           |
| ------------------------------------------------------------ | ------------------------------ |
| [troubleshooter.migrator.compare_ms_ckpt](api/migrator/compare_ms_ckpt.md) | 比较两个MindSpore的模型参数    |
| [troubleshooter.migrator.compare_pth_and_ckpt](api/migrator/compare_pth_and_ckpt.md) | 比较PyTorch和MindSpore模型参数 |

## 数据保存

| 接口名                             | 描述                                   |
| ---------------------------------- | -------------------------------------- |
| [troubleshooter.save](api/save.md) | 用于保存MindSpore和PyTorch的Tensor数据 |

## 数据比较

| 接口名                                                       | 描述                    |
| ------------------------------------------------------------ | ----------------------- |
| [troubleshooter.migrator.compare_npy_dir](api/migrator/compare_npy_dir.md) | 比较两个目录下的npy文件 |
| [troubleshooter.migrator.compare_grads_dir](api/migrator/compare_grads_dir.md) | 比较梯度                |

用于梯度比较获取**name_map_list**

| 接口名                                                       | 描述                                |
| ------------------------------------------------------------ | ----------------------------------- |
| [troubleshooter.migrator.get_name_map_list_by_name](api/migrator/get_name_map_list.md) | 根据name完全相同的规则获取匹配列表  |
| [troubleshooter.migrator.get_name_map_list_by_number](api/migrator/get_name_map_list.md) | 根据number顺序获取匹配列表          |
| [troubleshooter.migrator.get_name_map_list_by_shape_edit_distance](api/migrator/get_name_map_list.md) | 根据shape的最小编辑距离获取匹配列表 |

## 自动化比较

| 接口名                                                       | 描述                    |
| ------------------------------------------------------------ | ----------------------- |
| [troubleshooter.migrator.compare_npy_dir](api/migrator/compare_npy_dir.md) | 比较两个目录下的npy文件 |
| [troubleshooter.migrator.compare_grads_dir](api/migrator/compare_grads_dir.md) | 比较梯度                |
