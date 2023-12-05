## troubleshooter.widget.save_convert
>
> troubleshooter.widget.save_convert(file: str, output_path: str)

将`save`当`output_mode='print'`保存的文件转化为npy文件。

### 参数

- file: Ascend平台图模式下，使用ts.save output_mode='print'保存得到的文件。
- output_path: 输出文件夹路径，用于保存解析后的npy文件。

### 样例

```python
import troubleshooter as ts

# ts_save_file is the file saved by ts.save
ts.widget.save_convert(file='ts_save_file', output_path='npy_files')
```
