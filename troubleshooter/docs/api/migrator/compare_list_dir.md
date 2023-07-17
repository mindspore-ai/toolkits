## troubleshooter.migrator.compare_list_dir

> compare_list_dir(orig_dir, target_dir, rtol=1e-4, atol=1e-4, equal_nan=False, compare_shape=False)

批量对比两个目录下使用[save](api/save.md)保存的列表类型Tensor。

使用[troubleshooter.migrator.get_name_map_list_by_number](./get_name_map_list.md#troubleshootermigratorget_name_map_list_by_number)规则获取名称映射。

会计算`numpy.allclose`、`allclose`达标比例、余弦相似度、差异值的 $mean$ / $max$ 统计量等信息。

### 参数：

- orig_dir(`str`): 需要对比的npy文件所在的目录。
- target_dir(`str`): 目标数据所在的目录。
- rtol(`float`): 相对误差，默认值为`1e-4`，内部调用`numpy.allclose`的参数。
- atol(`float`): 绝对误差，默认值为`1e-4`，内部调用`numpy.allclose`的参数。
- equal_nan(`bool`): 是否将nan视为相等，默认值为 `False`，内部调用`numpy.allclose`的参数。
- compare_shape(`bool`): 是否比较shape信息，默认值False。
