# 网络调试小工具：

## 应用场景1：提供precision_tracker接口，根据传入的pb文件，识别图中算子执行后精度变化

### 接口定义
#### ```precision_tracker(abs_pb_filepath, precision_flags=('normal','raise','reduce'), **kwargs)```
根据传入的pb文件路径 `abs_pb_filepath` ，解析pb图结构，分析图中算子执行后精度变化，并生成csv文件。

**参数：**
- abs_pb_filepath：（`str`，必填），待分析的pb文件路径。
- precision_flags：（可迭代对象（`tuple` 或者 `list`），可选），算子精度标识，可选值有3个，支持多选，
  默认值：`('normal','raise','reduce')`。
  - normal：输出精度无变化的算子信息。
  - raise：输出精度上升的算子信息。
  - reduce：输出精度下降的算子信息。
- output_path：（`str`，可选）输出文件路径。默认值：与 `abs_pb_filepath` 路径一致。
- output_filename：（`str`，可选）输出文件名。默认值：`abs_pb_filepath` 的文件名加上‘.csv’后缀。

### 如何使用：
        # 导入precision_tracker接口
        from troubleshooter.toolbox import precision_tracker
        ......
        # 执行分析
        abs_pb_filepath = '/abs_path/xxx.pb'
        precision_tracker(abs_pb_filepath)
