## troubleshooter.migrator.save_net_and_weight_params

> troubleshooter.migrator.save_net_and_weight_params(net, path=os.getcwd(), weight_params_filename=None)

将网络对象保存成文件。

支持`MindSpore.nn.Cell`和`torch.nn.Module`，保存的文件包含模型信息（与用print打印model对象的内容相同）、模型权重参数，对于`torch.nn.Module`还会保存迁移到MindSpore的映射文件（使用[`get_weight_map`](get_weight_map.md)获得的文件）。

### 参数

- net: 需要保存信息的网络，支持`MindSpore.nn.Cell`和`torch.nn.Module`。
- path: 数据保存的路径，默认为当前路径。
- weight_params_filename: 模型权重的文件名，默认为`{torch/mindsproe}_troubleshooter_create.{pth/ckpt}`。
