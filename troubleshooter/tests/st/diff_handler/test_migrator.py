import pytest
import mindspore
import torch
import torch.nn as nn
import troubleshooter as ts
import torch.optim as optim
from collections import OrderedDict
from mindspore.common.initializer import Normal
from troubleshooter.migrator.weight_migrator import compare_pth_and_ckpt

class MyModule(nn.Module):
    def __init__(self, in_features, out_classes):
        super(MyModule, self).__init__()

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

class MSNet(mindspore.nn.Cell):
    def __init__(self, in_features, out_classes):
        super(MSNet, self).__init__()

        self.features = mindspore.nn.SequentialCell(
            OrderedDict([
                ('Linear_mm', mindspore.nn.Dense(in_features, 64, weight_init=Normal(0.02))),
                ('bn_mm', mindspore.nn.BatchNorm1d(64)),
                ('relu_mm', mindspore.nn.ReLU()),
                ('Linear_mm', mindspore.nn.Dense(64, out_classes, weight_init=Normal(0.02)))
            ])
        )

    def construct(self, x):
        x = self.features(x)
        return x

class MyNet(nn.Module):
    def __init__(self, in_features, out_classes):
        super(MyNet, self).__init__()

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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ordereddict_sequential_case(capsys):
    torch_net = MyModule(in_features=10, out_classes=2)
    torch.save(torch_net.state_dict(), "/tmp/torch_net.pth")
    pth_path = "/tmp/torch_net.pth"
    wm = ts.WeightMigrator(pt_model=torch_net, pth_file_path=pth_path, ckpt_save_path='/tmp/convert_resnet.ckpt')
    wm.convert()
    result = capsys.readouterr().out
    key_result = 'features.bn_mm.weight        |        features.bn_mm.gamma'
    assert result.count('True') == 4 and result.count(key_result) == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_save_model_pth_case(capsys):
    torch_net = MyModule(in_features=10,out_classes=2)
    #save model
    torch.save(torch_net, "/tmp/torch_net.pth")
    pth_path = "/tmp/torch_net.pth"
    wm = ts.WeightMigrator(pt_model=torch_net, pth_file_path=pth_path, ckpt_save_path='/tmp/convert_resnet.ckpt')
    wm.convert()
    result = capsys.readouterr().out
    key_result = 'features.bn_mm.weight        |        features.bn_mm.gamma'
    assert result.count('True') == 4 and result.count(key_result) == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_torch_modulelist_and_loadckpt_case(capsys):
    class MyNet_CellList(mindspore.nn.Cell):
        def __init__(self, in_channels, out_channels, hidden_size):
            super(MyNet_CellList, self).__init__()
            self.fc_layers = mindspore.nn.CellList()
            self.fc_layers.append(mindspore.nn.Dense(in_channels, hidden_size))
            self.fc_layers.append(mindspore.nn.Dense(hidden_size, out_channels))
            self.relu = mindspore.nn.ReLU()

        def construct(self, x):
            for i in range(len(self.fc_layers)):
                x = self.fc_layers[i](x)
                x = self.relu(x)

            return x
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

    torch_net=MyNet(in_channels=10,out_channels=2,hidden_size=20)
    ms_net = MyNet_CellList(in_channels=10, out_channels=2, hidden_size=20)
    torch.save(torch_net.state_dict(), "/tmp/torch_net.pth")
    pth_path = "/tmp/torch_net.pth"
    wm = ts.WeightMigrator(pt_model=torch_net, pth_file_path=pth_path, ckpt_save_path='/tmp/convert_resnet.ckpt')
    wm.convert(print_conv_info=False)
    param_dict = mindspore.load_checkpoint("/tmp/convert_resnet.ckpt")
    res = mindspore.load_param_into_net(ms_net, param_dict)
    ms_param_dict = ms_net.parameters_dict()
    assert len(ms_param_dict) == 4


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_modulelist_sequential_case(capsys):
    class MyModule(nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
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

    torch_net=MyModule()
    torch.save(torch_net.state_dict(), "/tmp/torch_net.pth")
    pth_path = "/tmp/torch_net.pth"
    wm = ts.WeightMigrator(pt_model=torch_net, pth_file_path=pth_path, ckpt_save_path='/tmp/convert_resnet.ckpt')
    wm.convert()
    result = capsys.readouterr().out
    key_result = 'features.0.weight   |        features.0.weight'
    assert result.count('False') == 20 and result.count(key_result) == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_weight_name_prefix_case(capsys):
    torch_net = MyModule(in_features=10,out_classes=2)
    torch.save(torch_net.state_dict(), "/tmp/torch_net.pth")
    pth_path = "/tmp/torch_net.pth"
    wm = ts.WeightMigrator(pt_model=torch_net, pth_file_path=pth_path, ckpt_save_path='/tmp/convert_resnet.ckpt')
    wm.convert(weight_name_prefix="pre_test")
    result = capsys.readouterr().out
    key_result = 'pre_test.features.Linear_mm.weight'
    assert result.count('pre_test') == 7 and result.count(key_result) == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_save_model_pth_and_input_dict_case(capsys):
    torch_net = MyModule(in_features=10,out_classes=2)
    #save model
    torch.save(torch_net, "/tmp/torch_net.pth")
    pth_path = "/tmp/torch_net.pth"
    model = torch.load(pth_path)
    # @验证模型场景下提取权重参数
    pd = model.state_dict()
    wm = ts.WeightMigrator(pt_model=torch_net, pth_para_dict=pd, ckpt_save_path='/tmp/convert_resnet.ckpt')
    wm.convert()
    result = capsys.readouterr().out
    key_result = 'features.bn_mm.weight        |        features.bn_mm.gamma'
    assert result.count('True') == 4 and result.count(key_result) == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_save_optimizer_case(capsys):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc = nn.Linear(10, 2)
        def forward(self, x):
            x = self.fc(x)
            return x

    model = Net()

    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 训练模型
    inputs = torch.randn(5, 10)
    labels = torch.randn(5, 2)
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, labels)
        loss.backward()
        optimizer.step()

    # 存储优化器参数
    torch.save(optimizer.state_dict(), 'optimizer.pth')

    # 加载优化器参数
    new_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    opt_para = torch.load('optimizer.pth')
    new_optimizer.load_state_dict(opt_para)
    pth_path = './optimizer.pth'
    try:
        wm = ts.WeightMigrator(pt_model=model, pth_file_path=pth_path, ckpt_save_path='./convert_resnet.ckpt')
        wm.convert()
    except ValueError as e:
        error_str = str(e)
        assert error_str.count('PTH file parsing failed, possible reasons:') == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_save_model_pth_and_input_dict_case(capsys):
    torch_net = MyModule(in_features=10, out_classes=2)
    #save model
    torch.save(torch_net, "/tmp/torch_net.pth")
    pth_path = "/tmp/torch_net.pth"
    model = torch.load(pth_path)
    # @验证模型场景下提取权重参数
    pd = model.state_dict()
    wm = ts.WeightMigrator(pt_model=torch_net, pth_para_dict=pd, ckpt_save_path='/tmp/convert_resnet.ckpt')
    wm.convert()
    result = capsys.readouterr().out
    key_result = 'features.bn_mm.weight        |        features.bn_mm.gamma'
    assert result.count('True') == 4 and result.count(key_result) == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_save_model_pth_and_input_dict_case(capsys):
    def custorm_weight_name(weight_name_map):
        prefix = '.custorm.'
        custorm_name_map = {}
        for key, value in weight_name_map.items():
            index = value.find(".")
            value = value[0:index] + prefix + value[index + 1:]
            #print(key, ":", value)
            custorm_name_map[key] = str(value)
        return custorm_name_map

    torch_net = MyModule(in_features=10, out_classes=2)
    #save model
    torch.save(torch_net, "/tmp/torch_net.pth")
    pth_path = "/tmp/torch_net.pth"
    # model = torch.load(pth_path)
    # @验证模型场景下提取权重参数
    # pd = model.state_dict()
    wm = ts.WeightMigrator(pt_model=torch_net, pth_file_path=pth_path, ckpt_save_path='/tmp/convert_resnet.ckpt')
    name_map, value_map = wm.get_weight_map(full_name_map=True)
    w_map = custorm_weight_name(name_map)
    # 将定制好的map传入转换接口
    # weight_map：传入定制后的map，以定制后的map进行权重转换
    wm.convert(weight_name_map=w_map)
    result = capsys.readouterr().out
    key_result = 'features.bn_mm.weight        |        features.custorm.bn_mm.gamma'
    assert result.count('.custorm.') == 7 and result.count(key_result) == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv1d_value_case(capsys):
    class MSNet(mindspore.nn.Cell):
        def __init__(self):
            super(MSNet, self).__init__()
            self.conv1d = mindspore.nn.Conv1d(256, 256, kernel_size=1, has_bias=True)

        def construct(self, A):
            return self.conv1d(A)

    class torchNet(torch.nn.Module):
        def __init__(self):
            super(torchNet, self).__init__()
            self.conv1d = torch.nn.Conv1d(256, 256, kernel_size=1)

        def forward(self, A):
            return self.conv1d(A)
    torch_net = torchNet()
    ms_net = MSNet()
    #save model
    torch.save(torch_net.state_dict(), "/tmp/torch_net.pth")
    pth_path = "/tmp/torch_net.pth"
    wm = ts.WeightMigrator(pt_model=torch_net, pth_file_path=pth_path, ckpt_save_path='/tmp/convert_resnet.ckpt')
    wm.convert()
    result = capsys.readouterr().out
    param_dict = mindspore.load_checkpoint("/tmp/convert_resnet.ckpt")
    res = mindspore.load_param_into_net(ms_net, param_dict)
    ms_param_dict = ms_net.parameters_dict()
    assert len(ms_param_dict) == 2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_compare_ckpt_value_case(capsys):
    torch_net = MyNet(in_features=10, out_classes=2)
    ms_net = MSNet(in_features=10, out_classes=2)
    torch.save(torch_net.state_dict(), "/tmp/torch_net.pth")
    ckpt_path = "/tmp/test.ckpt"
    mindspore.save_checkpoint(ms_net, ckpt_path)
    pth_path = "/tmp/torch_net.pth"
    wm = ts.WeightMigrator(pt_model=torch_net, pth_file_path=pth_path, ckpt_save_path='/tmp/convert_resnet.ckpt')
    wm.convert(print_conv_info=False)
    wm.compare_ckpt(ckpt_path=ckpt_path,
                    converted_ckpt_path='/tmp/convert_resnet.ckpt', compare_value=True)
    result = capsys.readouterr().out
    key_result = 'features.bn_mm.beta         |      features.bn_mm.beta       |           True'
    assert result.count(key_result) == 1



@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_compare_pth_value_case(capsys):
    torch_net = MyNet(in_features=10, out_classes=2)
    ms_net = MSNet(in_features=10, out_classes=2)
    torch.save(torch_net.state_dict(), "/tmp/torch_net.pth")
    ckpt_path = "/tmp/test.ckpt"
    mindspore.save_checkpoint(ms_net, ckpt_path)
    pth_path = "/tmp/torch_net.pth"
    wm = ts.WeightMigrator(pt_model=torch_net, pth_file_path=pth_path, ckpt_save_path='/tmp/convert_resnet.ckpt')
    wm.convert(print_conv_info=False)
    wm.compare_ckpt(ckpt_path=ckpt_path,
                    converted_ckpt_path='/tmp/convert_resnet.ckpt', compare_value=True, show_pth_name=True,
                    print_result=0)
    result = capsys.readouterr().out
    key_result = 'features.bn_mm.weight    |      features.bn_mm.gamma      |          True'
    assert result.count(key_result) == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_compare_pth_value_case(capsys):
    torch_net = MyNet(in_features=10, out_classes=2)
    ms_net = MSNet(in_features=10, out_classes=2)
    ckpt_path = "/tmp/test.ckpt"
    pth_path = "/tmp/torch_net.pth"
    torch.save(torch_net.state_dict(), pth_path)
    mindspore.save_checkpoint(ms_net, ckpt_path)
    compare_pth_and_ckpt(torch_net,pth_path,ckpt_path)
    result = capsys.readouterr().out
    key_result = 'features.bn_mm.weight    |      features.bn_mm.gamma      |          True'
    assert result.count(key_result) == 1

