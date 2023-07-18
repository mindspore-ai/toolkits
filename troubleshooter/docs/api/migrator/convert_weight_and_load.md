## troubleshooter.migrator.convert_weight_and_load

> troubleshooter.migrator.convert_weight_and_load(weight_map_path, pt_file_path, net)

根据权重映射文件，将PyTorch模型权重参数转换到MindSpore，并加载到MindSpore的net网络中。

### 参数

- weight_map_path：`troubleshooter.migrator.get_weight_map`生成的权重映射表路径。
- pt_file_path：PyTorch的pth文件路径。支持模型（例如：`torch.save(torch_net, "torch_net.pth")` ）和参数（例如：`torch.save(torch_net.state_dict(), "torch_net.pth"`)，两种形式pth文件的自动加载。
- net：待加载转换权重的MindSpore网络。
