"""PyTorch training"""
import torch
import torch.nn as nn
import troubleshooter as ts
from collections import OrderedDict
import mindspore
#@场景：验证模型场景下提取权重参数
class MyNet3(nn.Module):
    def __init__(self, in_features, out_classes):
        super(MyNet3, self).__init__()

        self.features = nn.Sequential(
        OrderedDict([
            ('Linear_mm', nn.Linear(in_features, 64)),
            ('bn_mm', nn.BatchNorm1d(64)),
            ('relu_mm', nn.ReLU()),
            ('Linear_mm', nn.Linear(64, out_classes))
        ])
        )

    def forward(self, x):
        x = self.features(x)
        return x

if __name__ == '__main__':


    #@验证：函数式转换
    #torch_net=MyModule1()

    #@验证：ParameterList权重参数
    #torch_net=MyModule2()

    #@验证：ModuleList+Sequential
    #torch_net=MyModule3()

    # @验证：ModuleList
    #torch_net = MyNet(in_channels=10,out_channels=2,hidden_size=20)

    # @验证：Sequential
    #torch_net = MyNet1(in_channels=3,out_channels=10,hidden_size=128)

    # @验证：Sequential+OrderedDict封装
    #torch_net = MyNet2(in_channels=3,out_channels=10,hidden_size=128)


    # @验证：Sequential + BatchNorm1d
    torch_net = MyNet3(in_features=10,out_classes=2)

    # @验证模型保存的权重迁移工具
    torch.save(torch_net, "torch_net.pth")
    pth_path = "./torch_net.pth"

    model = torch.load(pth_path)

    # @验证模型场景下提取权重参数
    #pd = model.state_dict()
    #for name, param in pd.items():
    #    print(name,":",param.size())

    wm = ts.weight_migrator(pt_model=torch_net, pth_file_path=pth_path, ckpt_save_path='./convert_resnet.ckpt')
    #w_maps = wm.get_weight_map(full_name_map=True, print_map=True)
    # test_map = {'bn1.bias': 'bn1.beta',}

    wm.convert(weight_name_prefix="miao")

    #打印ckpt
    #pth = "./convert_resnet.ckpt"
    #param_dict = mindspore.load_checkpoint(pth)
    #for key, value in param_dict.items():
    #    print(key, ':', value)