## troubleshooter.migrator.get_weight_map
> troubleshooter.migrator.get_weight_map(pt_net=None, weight_map_save_path=None, weight_name_prefix=None, custom_name_func=None, print_map=False)

生成PyTorch网络转换到MindSpore的参数映射文件。

常用于网络权重迁移、比较场景。

### 参数：

- pt_net：PyTorch网络实例。
- weight_map_save_path：转换后的权重映射表路径。
- weight_name_prefix：需要添加的权重前缀。
- custom_name_func：自定义名称映射函数。
- print_map：是否打印映射表。

