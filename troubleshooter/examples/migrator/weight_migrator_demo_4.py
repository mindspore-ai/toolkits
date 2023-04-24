"""PyTorch training"""
import torch
import torch.nn as nn
import troubleshooter as ts
from collections import OrderedDict
import mindspore
#@场景： ParameterList样例,保存到pth 价加载出来即变量名称+序号
# params_miao.0 : torch.Size([10, 10])
# params_miao.1 : torch.Size([10, 10])
# params_miao.2 : torch.Size([10, 10])
# params_miao.3 : torch.Size([10, 10])
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.params_miao = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(10, 10)) for i in range(10)])

    def forward(self, x):
        # ParameterList can act as an iterable, or be indexed using ints
        for i, p in enumerate(self.params):
            x = self.params[i // 2].mm(x) + p.mm(x)
        return x



#@场景： ParameterList样例,保存到pth 价加载出来即变量名称+序号
#params.0 : torch.Size([3, 4])
#params.1 : torch.Size([3])
#params.2 : torch.Size([4, 5])
#params.3 : torch.Size([4])
class MyModule2(torch.nn.Module):
    def __init__(self):
        super(MyModule2, self).__init__()
        self.params = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(3, 4), requires_grad=True),
            torch.nn.Parameter(torch.zeros(3), requires_grad=True),
            torch.nn.Parameter(torch.ones(4, 5), requires_grad=True),
            torch.nn.Parameter(torch.ones(4), requires_grad=True)
        ])

    def forward(self, x):
        weight1, bias1, weight2, bias2 = self.params

        x = torch.matmul(x, weight1) + bias1
        x = torch.nn.functional.relu(x)
        x = torch.matmul(x, weight2) + bias2
        return x


#@场景： 函数式调用 torch.nn.functional.batch_norm pth中不会保存直接在torch.nn.functional.batch_norm调用的参数
class MyModule1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.ones(3)
        self.bias = torch.zeros(3)
    def forward(self, x):

        result = torch.nn.functional.batch_norm(x, running_mean=None, running_var=None, weight=self.weight, bias=self.bias, training=True)
        return x



#@场景： ModuleList+Sequential
class MyModule3(nn.Module):
    def __init__(self):
        super(MyModule3, self).__init__()
        self.features = nn.ModuleList([
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU()
        ])

        self.classifier = nn.Sequential(
            nn.Linear(128 * 10 * 10, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = x.view(-1, 128 * 10 * 10)
        x = self.classifier(x)
        return x


#@场景： ModuleList
class MyNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size):
        super(MyNet, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(in_channels, hidden_size))
        self.fc_layers.append(nn.Linear(hidden_size, out_channels))

        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)
            x = self.relu(x)

        return x

#@场景： Sequential
# 打印结果：
# features.0.weight : torch.Size([16, 3, 3, 3])
# features.0.bias : torch.Size([16])
# features.2.weight : torch.Size([32, 16, 3, 3])
# features.2.bias : torch.Size([32])
# features.4.weight : torch.Size([64, 32, 3, 3])
# features.4.bias : torch.Size([64])
# classifier.0.weight : torch.Size([128, 64])
# classifier.0.bias : torch.Size([128])
# classifier.2.weight : torch.Size([10, 128])
# classifier.2.bias : torch.Size([10])
class MyNet1(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size):
        super(MyNet1, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(64, hidden_size),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_size, out_channels)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

# 场景：Sequential+OrderedDict封装
# features.0.weight : torch.Size([16, 3, 3, 3])
# features.0.bias : torch.Size([16])
# features.2.weight : torch.Size([32, 16, 3, 3])
# features.2.bias : torch.Size([32])
# features.4.weight : torch.Size([64, 32, 3, 3])
# features.4.bias : torch.Size([64])
# classifier.Linear_mm.weight : torch.Size([10, 128])
# classifier.Linear_mm.bias : torch.Size([10])
class MyNet2(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size):
        super(MyNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(OrderedDict([
                ('Linear_mm', nn.Linear(64, hidden_size)),
                ('fc_mm', nn.ReLU(inplace=True)),
                ('Linear_mm',nn.Linear(hidden_size, out_channels)),
            ]))
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

#@场景：Sequential + BatchNorm1d
class MyNet3(nn.Module):
    def __init__(self, in_features, out_classes):
        super(MyNet3, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, out_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return x

#@场景：Sequential + BatchNorm1d + OrderedDict封装
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

def custorm_weight_name_prefix(weight_name_map, prefix=None):
    if prefix:
        custorm_name_map = {}
        for key, value in weight_name_map.items():
            print(key, ":", prefix + '.' + value)
            custorm_name_map[key] = str(prefix) + '.' + str(value)
        return custorm_name_map
    else:
        return weight_name_map
def custorm_weight_name(weight_name_map):
    prefix='.custorm.'
    custorm_name_map = {}
    for key, value in weight_name_map.items():
        index = value.find(".")
        value = value[0:index] + prefix + value[index+1:]
        print(key, ":", value)
        custorm_name_map[key] = str(value)
    return custorm_name_map

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

    torch.save(torch_net.state_dict(), "torch_net.pth")

    # @验证模型保存
    # torch.save(torch_net, "torch_net.pth")
    pth_path = "./torch_net.pth"

    model = torch.load(pth_path)

    # @验证模型场景下提取权重参数
    #pd = model.state_dict()

    #for name, param in model.items():
    #    print(name,":",param.size())

    wm = ts.weight_migrator(pt_model=torch_net, pth_file_path=pth_path, ckpt_save_path='./convert_resnet.ckpt')
    #w_maps = wm.get_weight_map(full_name_map=True, print_map=True)
    # test_map = {'bn1.bias': 'bn1.beta',}

    # 用户可封装定制函数，例如：custorm_weight_name，然后通过修改w_map内容，完成映射关系的定制
    #name_map, value_map = wm.get_weight_map(full_name_map=True)
    #w_map = custorm_weight_name(name_map)
    # 将定制好的map传入转换接口
    # weight_map：传入定制后的map，以定制后的map进行权重转换
    #wm.convert(weight_name_map=w_map)

    wm.convert(weight_name_prefix="miao")

    #打印ckpt
    #pth = "./convert_resnet.ckpt"
    #param_dict = mindspore.load_checkpoint(pth)
    #for key, value in param_dict.items():
    #    print(key, ':', value)