## troubleshooter.migrator.convert_weight

> troubleshooter.migrator.convert_weight(weight_map_path=None, pt_file_path=None, ms_file_save_path=None)

转换PyTorch的模型参数(pth)为MindSpore模型参数(ckpt)

### 参数

- weight_map_path：`troubleshooter.migrator.get_weight_map`生成的权重映射文件路径。
- pt_file_path：PyTorch的pth文件路径。支持模型（例如：`torch.save(torch_net, "torch_net.pth")` ）和参数（例如：`torch.save(torch_net.state_dict(), "torch_net.pth"`)，两种形式pth文件的自动加载。
- ms_file_save_path：转换后的MindSpore的ckpt文件路径。

### kwargs参数

- pt_param_dict：当保存的pth文件内容经过定制时，则不能通过`pt_file_path`参数加载，可以通过此参数直接传入加载并解析后的权重参数字典。
- print_conv_info：是否打印转换信息。默认值：`True`。

### 样例

权重迁移场景参考[pth到ckpt权重自动转换](../../migrator.md#应用场景1pth到ckpt权重自动转换)
