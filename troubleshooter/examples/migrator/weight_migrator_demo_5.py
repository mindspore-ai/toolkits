import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, Parameter, ParameterTuple
from mindspore.common.initializer import Normal
import mindspore.ops.operations as P
from collections import OrderedDict
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        # @场景1  参数名称为设置的名称,设置名称的优先级，高于变量名称
        # 保存到ckpt加载后打印结果如下，会打印设置的参数名称 weightaaa，而不是变量名称weight1
        # weightaaa : Parameter (name=weightaaa, shape=(3, 4), dtype=Float64, requires_grad=True)
        # biasbbb : Parameter (name=biasbbb, shape=(3,), dtype=Float64, requires_grad=True)
        # weight2 : Parameter (name=weight2, shape=(4, 5), dtype=Float64, requires_grad=True)
        # bias2 : Parameter (name=bias2, shape=(4,), dtype=Float64, requires_grad=True)

        self.weight1 = Parameter(Tensor(np.ones([3, 4])), name="weightaaa")
        self.bias1 = Parameter(Tensor(np.zeros([3])), name="biasbbb")
        self.weight2 = Parameter(Tensor(np.ones([4, 5])), name="weight2")
        self.bias2 = Parameter(Tensor(np.zeros([4])), name="bias2")
        self.params = ParameterTuple((self.weight1, self.bias1, self.weight2, self.bias2))

        # @场景1-1  没设置名称，会按照变量的名称打印
        # 保存到ckpt加载后打印结果如下
        # weight1 : Parameter (name=weight1, shape=(3, 4), dtype=Float64, requires_grad=True)
        # bias1 : Parameter (name=bias1, shape=(3,), dtype=Float64, requires_grad=True)
        # weight2 : Parameter (name=weight2, shape=(4, 5), dtype=Float64, requires_grad=True)
        # bias2 : Parameter (name=bias2, shape=(4,), dtype=Float64, requires_grad=True)
        #self.weight1 = Parameter(Tensor(np.ones([3, 4])))
        #self.bias1 = Parameter(Tensor(np.zeros([3])))
        #self.weight2 = Parameter(Tensor(np.ones([4, 5])))
        #self.bias2 = Parameter(Tensor(np.zeros([4])))
        #self.params = ParameterTuple((self.weight1, self.bias1, self.weight2, self.bias2))

        # @场景2  参数名称为name设置的名称
        # 保存到ckpt加载后打印结果如下，
        # weight1 : Parameter (name=weight1, shape=(3, 4), dtype=Float64, requires_grad=True)
        # bias1 : Parameter (name=bias1, shape=(3,), dtype=Float64, requires_grad=True)
        # weight2 : Parameter (name=weight2, shape=(4, 5), dtype=Float64, requires_grad=True)
        # bias2 : Parameter (name=bias2, shape=(4,), dtype=Float64, requires_grad=True)        #
        #
        #self.params = ParameterTuple((Parameter(Tensor(np.ones([3, 4])), name="weight1"),
        #                              Parameter(Tensor(np.zeros([3])), name="bias1"),
        #                              Parameter(Tensor(np.ones([4, 5])), name="weight2"),
        #                              Parameter(Tensor(np.zeros([4])), name="bias2")))
        # @场景3  会报参数重名错误
        #self.params = ParameterTuple((Parameter(Tensor(np.ones([3, 4]))),
        #                              Parameter(Tensor(np.zeros([3]))),
        #                              Parameter(Tensor(np.ones([4, 5]))),
        #                              Parameter(Tensor(np.zeros([4])))))
    def construct(self, x):
        x = nn.Dense(*self.params[:2])(x)
        x = nn.ReLU()(x)
        x = nn.Dense(*self.params[2:])(x)
        return x


#@场景： CellList
# 0.weight : Parameter (name=0.weight, shape=(20, 10), dtype=Float32, requires_grad=True)
# 0.bias : Parameter (name=0.bias, shape=(20,), dtype=Float32, requires_grad=True)
# 1.weight : Parameter (name=1.weight, shape=(2, 20), dtype=Float32, requires_grad=True)
# 1.bias : Parameter (name=1.bias, shape=(2,), dtype=Float32, requires_grad=True)
class MyNet_CellList(nn.Cell):
    def __init__(self, in_channels, out_channels, hidden_size):
        super(MyNet_CellList, self).__init__()
        self.fc_layers = nn.CellList()
        self.fc_layers.append(nn.Dense(in_channels, hidden_size))
        self.fc_layers.append(nn.Dense(hidden_size, out_channels))

        self.relu = nn.ReLU()

    def construct(self, x):
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)
            x = self.relu(x)

        return x



# 验证SequentialCell
# 打印结果
# features.0.weight : Parameter (name=features.0.weight, shape=(16, 3, 3, 3), dtype=Float32, requires_grad=True)
# features.0.bias : Parameter (name=features.0.bias, shape=(16,), dtype=Float32, requires_grad=True)
# features.2.weight : Parameter (name=features.2.weight, shape=(32, 16, 3, 3), dtype=Float32, requires_grad=True)
# features.2.bias : Parameter (name=features.2.bias, shape=(32,), dtype=Float32, requires_grad=True)
# features.4.weight : Parameter (name=features.4.weight, shape=(64, 32, 3, 3), dtype=Float32, requires_grad=True)
# features.4.bias : Parameter (name=features.4.bias, shape=(64,), dtype=Float32, requires_grad=True)
# classifier.0.weight : Parameter (name=classifier.0.weight, shape=(128, 64), dtype=Float32, requires_grad=True)
# classifier.0.bias : Parameter (name=classifier.0.bias, shape=(128,), dtype=Float32, requires_grad=True)
# classifier.2.weight : Parameter (name=classifier.2.weight, shape=(10, 128), dtype=Float32, requires_grad=True)
# classifier.2.bias : Parameter (name=classifier.2.bias, shape=(10,), dtype=Float32, requires_grad=True)
class MyNet(nn.Cell):
    def __init__(self, in_channels, out_channels, hidden_size):
        super(MyNet, self).__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.SequentialCell(
            nn.Dense(64, hidden_size),
            nn.ReLU(),
            nn.Dense(hidden_size, out_channels)
        )

    def construct(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = P.Flatten()(x)
        x = self.classifier(x)
        return x

# 验证SequentialCell+OrderedDict
# 打印结果：
# features.0.weight : Parameter (name=features.0.weight, shape=(16, 3, 3, 3), dtype=Float32, requires_grad=True)
# features.0.bias : Parameter (name=features.0.bias, shape=(16,), dtype=Float32, requires_grad=True)
# features.2.weight : Parameter (name=features.2.weight, shape=(32, 16, 3, 3), dtype=Float32, requires_grad=True)
# features.2.bias : Parameter (name=features.2.bias, shape=(32,), dtype=Float32, requires_grad=True)
# features.4.weight : Parameter (name=features.4.weight, shape=(64, 32, 3, 3), dtype=Float32, requires_grad=True)
# features.4.bias : Parameter (name=features.4.bias, shape=(64,), dtype=Float32, requires_grad=True)
# classifier.Dense_mm.weight : Parameter (name=classifier.Dense_mm.weight, shape=(10, 128), dtype=Float32, requires_grad=True)
# classifier.Dense_mm.bias : Parameter (name=classifier.Dense_mm.bias, shape=(10,), dtype=Float32, requires_grad=True)
class MyNet(nn.Cell):
    def __init__(self, in_channels, out_channels, hidden_size):
        super(MyNet, self).__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.SequentialCell(
            OrderedDict([
                ('Dense_mm', nn.Dense(64, hidden_size)),
                ('ReLUmm', nn.ReLU()),
                ('Dense_mm', nn.Dense(hidden_size, out_channels)),
            ])
        )

    def construct(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = P.Flatten()(x)
        x = self.classifier(x)
        return x

# 验证SequentialCell+BatchNorm1d
# features.0.weight : Parameter (name=features.0.weight, shape=(64, 10), dtype=Float32, requires_grad=True)
# features.0.bias : Parameter (name=features.0.bias, shape=(64,), dtype=Float32, requires_grad=True)
# features.1.moving_mean : Parameter (name=features.1.moving_mean, shape=(64,), dtype=Float32, requires_grad=True)
# features.1.moving_variance : Parameter (name=features.1.moving_variance, shape=(64,), dtype=Float32, requires_grad=True)
# features.1.gamma : Parameter (name=features.1.gamma, shape=(64,), dtype=Float32, requires_grad=True)
# features.1.beta : Parameter (name=features.1.beta, shape=(64,), dtype=Float32, requires_grad=True)
# features.3.weight : Parameter (name=features.3.weight, shape=(2, 64), dtype=Float32, requires_grad=True)
# features.3.bias : Parameter (name=features.3.bias, shape=(2,), dtype=Float32, requires_grad=True)
class MyNet(nn.Cell):
    def __init__(self, in_features, out_classes):
        super(MyNet, self).__init__()

        self.features = nn.SequentialCell([
            nn.Dense(in_features, 64, weight_init=Normal(0.02)),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dense(64, out_classes, weight_init=Normal(0.02))
        ])

    def construct(self, x):
        x = self.features(x)
        return x

# 验证SequentialCell+BatchNorm1d+OrderedDict
class MyNet(nn.Cell):
    def __init__(self, in_features, out_classes):
        super(MyNet, self).__init__()

        self.features = nn.SequentialCell(
        OrderedDict([
            ('Linear_mm', nn.Dense(in_features, 64, weight_init=Normal(0.02))),
            ('bn_mm', nn.BatchNorm1d(64)),
            ('relu_mm', nn.ReLU()),
            ('Linear_mm', nn.Dense(64, out_classes, weight_init=Normal(0.02)))
        ])
        )
    def construct(self, x):
        x = self.features(x)
        return x


#@验证1：ParameterTuple
#net = Net()

#@验证2：CellList
net = MyNet_CellList(in_channels=10,out_channels=2, hidden_size=20)

#@验证3：SequentialCell
#net = MyNet(in_channels=3, out_channels=10, hidden_size=128)

#@验证4：SequentialCell BatchNorm1d
#net = MyNet(in_features=10, out_classes=2)

#@验证5 SequentialCell+BatchNorm1d+OrderedDict
#net = MyNet(in_features=10, out_classes=2)

pth = "./test.ckpt"
mindspore.save_checkpoint(net, pth)
param_dict = mindspore.load_checkpoint(pth)
for key, value in param_dict.items():
    print(key, ':', value)

