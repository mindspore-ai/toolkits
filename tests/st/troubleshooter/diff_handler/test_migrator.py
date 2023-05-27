import os
import re
from collections import OrderedDict

import mindspore
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import troubleshooter as ts
from mindspore.common.initializer import Normal


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
                ('Linear_mm', mindspore.nn.Dense(
                    in_features, 64, weight_init=Normal(0.02))),
                ('bn_mm', mindspore.nn.BatchNorm1d(64)),
                ('relu_mm', mindspore.nn.ReLU()),
                ('Linear_mm', mindspore.nn.Dense(
                    64, out_classes, weight_init=Normal(0.02)))
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
    torch_path = "/tmp/torch_net_ordereddict_sequential.pth"
    map_file_path = "/tmp/torch_net_ordereddict_sequential_map.json"
    ms_path = "/tmp/convert_ordereddict_sequential.ckpt"
    torch.save(torch_net.state_dict(), torch_path)
    ts.migrator.get_weight_map(pt_model=torch_net,
                               weight_map_save_path=map_file_path,
                               print_map=True)
    ts.migrator.convert_weight(weight_map_path=map_file_path,
                               pt_file_path=torch_path,
                               ms_file_save_path=ms_path)

    result = capsys.readouterr().out
    key_result = 'features.bn_mm.weight        |        features.bn_mm.gamma'
    os.remove(torch_path)
    os.remove(map_file_path)
    os.remove(ms_path)
    assert result.count('True') == 4 and result.count(key_result) == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_save_model_pth_case(capsys):
    torch_net = MyModule(in_features=10, out_classes=2)
    # save model
    pth_path = "/tmp/torch_net.pth"
    ms_file_path = '/tmp/convert_resnet.ckpt'
    map_file_path = "/tmp/torch_net_map.json"
    torch.save(torch_net, pth_path)
    ts.migrator.get_weight_map(pt_model=torch_net,
                               weight_map_save_path=map_file_path,
                               print_map=True)
    ts.migrator.convert_weight(weight_map_path=map_file_path,
                               pt_file_path=pth_path,
                               ms_file_save_path=ms_file_path)
    result = capsys.readouterr().out
    key_result = 'features.bn_mm.weight        |        features.bn_mm.gamma'
    os.remove(map_file_path)
    os.remove(pth_path)
    os.remove(ms_file_path)
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
            self.fc_layers.append(
                mindspore.nn.Dense(hidden_size, out_channels))
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

    pth_path = "/tmp/torch_modulelist_net.pth"
    ms_file_path = '/tmp/convert_modulelist_resnet.ckpt'
    map_file_path = "/tmp/torch_modulelist_net_map.json"
    torch_net = MyNet(in_channels=10, out_channels=2, hidden_size=20)
    ms_net = MyNet_CellList(in_channels=10, out_channels=2, hidden_size=20)
    torch.save(torch_net.state_dict(), pth_path)
    ts.migrator.get_weight_map(pt_model=torch_net,
                               weight_map_save_path=map_file_path,
                               print_map=True)
    ts.migrator.convert_weight(weight_map_path=map_file_path,
                               pt_file_path=pth_path,
                               ms_file_save_path=ms_file_path)
    param_dict = mindspore.load_checkpoint(ms_file_path)
    res = mindspore.load_param_into_net(ms_net, param_dict)
    ms_param_dict = ms_net.parameters_dict()
    os.remove(map_file_path)
    os.remove(pth_path)
    os.remove(ms_file_path)
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

    pth_path = "/tmp/torch_modulelist_sequential_net.pth"
    ms_file_path = '/tmp/convert_modulelist_sequential_resnet.ckpt'
    map_file_path = "/tmp/torch_modulelist_sequential_net_map.json"
    torch_net = MyModule()
    torch.save(torch_net.state_dict(), pth_path)
    ts.migrator.get_weight_map(pt_model=torch_net,
                               weight_map_save_path=map_file_path,
                               print_map=True)
    ts.migrator.convert_weight(weight_map_path=map_file_path,
                               pt_file_path=pth_path,
                               ms_file_save_path=ms_file_path)
    result = capsys.readouterr().out
    key_result = 'features.0.weight    |      features.0.weight'
    os.remove(map_file_path)
    os.remove(pth_path)
    os.remove(ms_file_path)
    assert result.count('False') == 20 and result.count(key_result) == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_weight_name_prefix_case(capsys):
    pth_path = "/tmp/torch_weight_name_prefix_net.pth"
    ms_file_path = '/tmp/convert_weight_name_prefix_resnet.ckpt'
    map_file_path = "/tmp/torch_weight_name_prefix_net_map.json"
    torch_net = MyModule(in_features=10, out_classes=2)
    torch.save(torch_net.state_dict(), pth_path)
    ts.migrator.get_weight_map(pt_model=torch_net,
                               weight_map_save_path=map_file_path,
                               weight_name_prefix="pre_test")
    ts.migrator.convert_weight(weight_map_path=map_file_path,
                               pt_file_path=pth_path,
                               ms_file_save_path=ms_file_path,
                               )
    result = capsys.readouterr().out
    key_result = 'pre_test.features.Linear_mm.weight'
    os.remove(map_file_path)
    os.remove(pth_path)
    os.remove(ms_file_path)
    assert result.count('pre_test') == 7 and result.count(key_result) == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_save_model_pth_and_input_dict_case(capsys):
    pth_path = "/tmp/torch_dict_net.pth"
    ms_file_path = '/tmp/convert_dict_resnet.ckpt'
    map_file_path = "/tmp/torch_dict_net_map.json"
    torch_net = MyModule(in_features=10, out_classes=2)
    # save model
    torch.save(torch_net, pth_path)
    model = torch.load(pth_path)
    # @验证模型场景下提取权重参数
    pd = model.state_dict()
    ts.migrator.get_weight_map(pt_model=torch_net,
                               weight_map_save_path=map_file_path,
                               print_map=True)
    ts.migrator.convert_weight(weight_map_path=map_file_path,
                               pt_param_dict=pd,
                               ms_file_save_path=ms_file_path)
    result = capsys.readouterr().out
    key_result = 'features.bn_mm.weight        |        features.bn_mm.gamma'
    os.remove(map_file_path)
    os.remove(pth_path)
    os.remove(ms_file_path)
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
    pth_path = '/tmp/optimizer.pth'
    ms_file_path = '/tmp/convert_optimizer_resnet.ckpt'
    map_file_path = "/tmp/torch_optimizer_net_map.json"
    torch.save(optimizer.state_dict(), pth_path)

    # 加载优化器参数
    new_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    opt_para = torch.load(pth_path)
    new_optimizer.load_state_dict(opt_para)
    try:
        # wm = ts.WeightMigrator(pt_model=model, pth_file_path=pth_path, ckpt_save_path='./convert_resnet.ckpt')
        # wm.convert()
        ts.migrator.get_weight_map(pt_model=model,
                                   weight_map_save_path=map_file_path,
                                   print_map=True)
        ts.migrator.convert_weight(weight_map_path=map_file_path,
                                   pt_file_path=pth_path,
                                   ms_file_save_path=ms_file_path)
        os.remove(map_file_path)
        os.remove(pth_path)
        os.remove(ms_file_path)
    except ValueError as e:
        error_str = str(e)
        assert error_str.count(
            'PTH file parsing failed, possible reasons:') == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_custorm_weight_case(capsys):
    def custorm_weight_name(weight_name_map):
        prefix = '.custorm.'
        custorm_name_map = {}
        for key, value in weight_name_map.items():
            index = value.find(".")
            value = value[0:index] + prefix + value[index + 1:]
            # print(key, ":", value)
            custorm_name_map[key] = str(value)
        return custorm_name_map

    pth_path = "/tmp/torch_custorm_weight_net.pth"
    ms_file_path = '/tmp/convert_custorm_weight_resnet.ckpt'
    map_file_path = "/tmp/torch_custorm_weight_net_map.json"
    torch_net = MyModule(in_features=10, out_classes=2)
    # save model
    torch.save(torch_net, pth_path)
    # model = torch.load(pth_path)
    # @验证模型场景下提取权重参数
    # pd = model.state_dict()
    # wm = ts.WeightMigrator(pt_model=torch_net, pth_file_path=pth_path, ckpt_save_path='/tmp/convert_resnet.ckpt')
    # name_map, value_map = wm.get_weight_map(full_name_map=True)

    ts.migrator.get_weight_map(pt_model=torch_net,
                               weight_map_save_path=map_file_path,
                               custom_name_func=custorm_weight_name)
    # 将定制好的map传入转换接口
    # weight_map：传入定制后的map，以定制后的map进行权重转换
    # wm.convert(weight_name_map=w_map)
    ts.migrator.convert_weight(weight_map_path=map_file_path,
                               pt_file_path=pth_path,
                               ms_file_save_path=ms_file_path)
    result = capsys.readouterr().out
    key_result = 'features.bn_mm.weight        |        features.custorm.bn_mm.gamma'
    os.remove(map_file_path)
    os.remove(pth_path)
    os.remove(ms_file_path)
    assert result.count('.custorm.') == 7 and result.count(key_result) == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv1d_value_case(capsys):
    class MSNet(mindspore.nn.Cell):
        def __init__(self):
            super(MSNet, self).__init__()
            self.conv1d = mindspore.nn.Conv1d(
                256, 256, kernel_size=1, has_bias=True)

        def construct(self, A):
            return self.conv1d(A)

    class torchNet(torch.nn.Module):
        def __init__(self):
            super(torchNet, self).__init__()
            self.conv1d = torch.nn.Conv1d(256, 256, kernel_size=1)

        def forward(self, A):
            return self.conv1d(A)

    pth_path = "/tmp/torch_conv1d_value_net.pth"
    ms_file_path = '/tmp/convert_conv1d_value_resnet.ckpt'
    map_file_path = "/tmp/torch_conv1d_value_net_map.json"
    torch_net = torchNet()
    ms_net = MSNet()
    # save model
    torch.save(torch_net.state_dict(), pth_path)
    # wm = ts.WeightMigrator(pt_model=torch_net, pth_file_path=pth_path, ckpt_save_path='/tmp/convert_resnet.ckpt')
    # wm.convert()
    ts.migrator.get_weight_map(pt_model=torch_net,
                               weight_map_save_path=map_file_path,
                               print_map=True)
    ts.migrator.convert_weight(weight_map_path=map_file_path,
                               pt_file_path=pth_path,
                               ms_file_save_path=ms_file_path)
    result = capsys.readouterr().out
    param_dict = mindspore.load_checkpoint(ms_file_path)
    res = mindspore.load_param_into_net(ms_net, param_dict)
    ms_param_dict = ms_net.parameters_dict()
    os.remove(map_file_path)
    os.remove(pth_path)
    os.remove(ms_file_path)
    assert len(ms_param_dict) == 2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_compare_ckpt_value_case(capsys):
    ckpt_path = "/tmp/compare_test.ckpt"
    pth_path = "/tmp/torch_compare_net.pth"
    ms_file_path = '/tmp/convert_compare_resnet.ckpt'
    map_file_path = "/tmp/torch_compare_net_map.json"
    torch_net = MyNet(in_features=10, out_classes=2)
    ms_net = MSNet(in_features=10, out_classes=2)
    torch.save(torch_net.state_dict(), pth_path)
    mindspore.save_checkpoint(ms_net, ckpt_path)
    # wm = ts.WeightMigrator(pt_model=torch_net, pth_file_path=pth_path, ckpt_save_path='/tmp/convert_resnet.ckpt')
    # wm.convert(print_conv_info=False)
    ts.migrator.get_weight_map(pt_model=torch_net,
                               weight_map_save_path=map_file_path,
                               print_map=True)
    ts.migrator.convert_weight(weight_map_path=map_file_path,
                               pt_file_path=pth_path,
                               ms_file_save_path=ms_file_path)
    # wm.compare_ckpt(ckpt_path=ckpt_path,
    #               converted_ckpt_path='/tmp/convert_resnet.ckpt', compare_value=True)
    ts.migrator.compare_ms_ckpt(orig_file_path=ms_file_path,
                                target_file_path=ckpt_path,
                                compare_value=True)
    result = capsys.readouterr().out
    key_result = 'features.bn_mm.beta         |      features.bn_mm.beta       |           True'
    os.remove(ckpt_path)
    os.remove(pth_path)
    os.remove(ms_file_path)
    os.remove(map_file_path)
    assert result.count(key_result) == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_compare_show_pth_name_case(capsys):
    ckpt_path = "/tmp/compare_show_test.ckpt"
    pth_path = "/tmp/torch_compare_show_net.pth"
    ms_file_path = '/tmp/convert_compare_show_resnet.ckpt'
    map_file_path = "/tmp/torch_compare_show_net_map.json"
    torch_net = MyNet(in_features=10, out_classes=2)
    ms_net = MSNet(in_features=10, out_classes=2)
    torch.save(torch_net.state_dict(), pth_path)
    mindspore.save_checkpoint(ms_net, ckpt_path)

    ts.migrator.get_weight_map(pt_model=torch_net,
                               weight_map_save_path=map_file_path,
                               print_map=True)
    ts.migrator.convert_weight(weight_map_path=map_file_path,
                               pt_file_path=pth_path,
                               ms_file_save_path=ms_file_path, print_level=0)
    ts.migrator.compare_ms_ckpt(orig_file_path=ms_file_path,
                                target_file_path=ckpt_path,
                                compare_value=True,
                                weight_map_path=map_file_path)
    os.remove(ckpt_path)
    os.remove(pth_path)
    os.remove(ms_file_path)
    os.remove(map_file_path)
    result = capsys.readouterr().out
    key_result = 'features.bn_mm.weight        |        features.bn_mm.gamma        |              True'
    assert result.count(key_result) == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_compare_pth_value_case(capsys):
    ckpt_path = "/tmp/compare_ckpt_net.ckpt"
    pth_path = "/tmp/compare_torch_net.pth"
    map_file_path = "/tmp/compare_torch_net_map.json"
    torch_net = MyNet(in_features=10, out_classes=2)
    ms_net = MSNet(in_features=10, out_classes=2)
    torch.save(torch_net.state_dict(), pth_path)
    mindspore.save_checkpoint(ms_net, ckpt_path)
    ts.migrator.get_weight_map(
        pt_model=torch_net, weight_map_save_path=map_file_path, print_map=True)
    ts.migrator.compare_pth_and_ckpt(weight_map_path=map_file_path,
                                     pt_file_path=pth_path,
                                     ms_file_path=ckpt_path)

    result = capsys.readouterr().out
    pattern = r"features\.bn_mm\.weight[ \t]+\|[ \t]+features\.bn_mm\.gamma[ \t]+\|[ \t]+True"
    match = re.search(pattern, result)
    os.remove(ckpt_path)
    os.remove(map_file_path)
    os.remove(pth_path)
    assert match is not None
