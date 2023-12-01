## troubleshooter.save_grad
>
> troubleshooter.save_grad(file:str, data:Tensor, suffix='backward', use_print=False)

`MindSpore`和`PyTorch`的统一反向数据保存接口，会将`Tensor`对应的反向梯度保存为numpy的npy格式文件。

### 参数

- file: 文件名路径。为避免同名文件覆盖，保存的文件名称会自动添加前缀，前缀从0开始，按照执行顺序递增。
- data: 数据，支持保存`Tensor`对应的反向梯度。
- suffix: 文件名后缀，默认值为`backward`。
- use_print：是否使用print进行数据输出。print默认会输出到屏幕，MindSpore图模式下可以设置context中的print_file_path输出到文件，输出到文件后可以使用save_convert接口解析为npy文件。

### 返回
- Tensor，与输入数据相同。**返回值需要传递给下一个API的输入，以构成反向图从而保存反向数据**。

### 文件名称格式

存储的文件名称为 **[id]\_name\_[suffix].npy**。
- id为按照执行顺序递增的前缀；
- name为file中指定的文件名；
- suffix为指定的后缀，默认为`backward`。

> **Warning:**
>
> - 在MindSpore 2.3之前的版本中，save_grad只支持使用print输出。
>
> - MindSpore 中数据保存是异步处理的。当数据量过大或主进程退出过快时，可能出现数据丢失的问题，需要主动控制主进程销毁时间，例如使用sleep。


### 样例
