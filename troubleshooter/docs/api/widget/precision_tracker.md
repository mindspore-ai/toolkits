## troubleshooter.widget.precision_tracker 

> troubleshooter.widget.precision_tracker(abs_pb_filepath, precision_flags=('normal','raise','reduce'), **kwargs)

根据传入的pb文件路径 `abs_pb_filepath` ，解析pb图结构，分析图中算子执行后精度变化，并生成csv文件。

### 参数
- abs_pb_filepath （`str`，必填）：待分析的pb文件路径。
- precision_flags （可迭代对象（`tuple` 或者 `list`），可选）：算子精度标识，可选值有3个，支持多选，
  默认值：`('normal','raise','reduce')`。
  - normal：输出精度无变化的算子信息。
  - raise：输出精度上升的算子信息。
  - reduce：输出精度下降的算子信息。
- output_path （`str`，可选）：输出文件路径。默认值：与 `abs_pb_filepath` 路径一致。
- output_filename （`str`，可选）：输出文件名。默认值：`abs_pb_filepath` 的文件名加上.csv后缀。

具体使用方法请查看：[应用场景1：根据保存的pb文件，识别图中算子执行后精度变化](../../widget.md)
