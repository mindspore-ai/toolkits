# get_name_map_list
为方便获取文件的映射结果，TroubleShooter提供了多种策略函数获取`name_map_list`。

## troubleshooter.migrator.get_name_map_list_by_name
> troubleshooter.migrator.get_name_map_list_by_name(orig_dir, target_dir)

名称完全一致时匹配。

### 参数

- orig_dir：源目录。
- target_dir：目标目录。

### 返回值：

- list[tuple]：匹配的文件列表。

## troubleshooter.migrator.get_name_map_list_by_number

> get_name_map_list_by_number(orig_dir, target_dir)

按照文件名中的首尾数字排升序，之后顺序做配对。

### 参数

- orig_dir：源目录。
- target_dir：目标目录。

### 返回值：

- list[tuple]：匹配的文件列表。

## troubleshooter.migrator.get_name_map_list_by_shape_edit_distance

> troubleshooter.migrator.get_name_map_list_by_shape_edit_distance(orig_dir, target_dir, *, del_cost=1, ins_cost=1, rep_cost=5)

按照文件名中的数字排序后，根据shape信息计算最小编辑距离计算进行匹配。

常用于结构不完全一致时的比较，例如梯度比较。

### 参数

- orig_dir：源目录。
- target_dir：目标目录。
- del_cost：编辑距离匹配时删除代价。默认值：1。
- ins_cost：编辑距离匹配时插入代价。默认值：1。
- rep_cost：编辑距离匹配时替换代价。默认值：5。

### 返回值：

- list[tuple]：匹配的文件列表。

以下为使用`ts.save`对不等长的数据进行连续保存，分别使用三种不同匹配算法获得的匹配效果。
```python
data0 = mindspore.ops.randn((2,3))
data1 = mindspore.ops.randn((3,5))
data2 = mindspore.ops.randn((5,5))
path1 = tempfile.mkdtemp(prefix='ta')
path2 = tempfile.mkdtemp(prefix='tb')
>
ts.save(os.path.join(path1, 'data'), [data0, data1, data2])
ts.save(os.path.join(path2, 'data'), [data0, data2])
>
by_name = ts.migrator.get_name_map_list_by_name(path1, path2)
by_number = ts.migrator.get_name_map_list_by_number(path1, path2)
by_edit_shape = ts.migrator.get_name_map_list_by_shape_edit_distance(path1, path2)
```
![get_name_list](../../images/get_map_name_list.png)