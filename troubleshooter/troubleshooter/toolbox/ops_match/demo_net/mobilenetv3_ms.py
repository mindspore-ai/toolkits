import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from mindspore.common.tensor import Tensor
import wrap_nn
from hooks import acc_cmp_dump, set_dump_switch
from initialize import register_hook

wrap_nn.wrap_nn_cell_and_bind()
ms.set_context(mode=ms.PYNATIVE_MODE)
# class HSwish(nn.Cell):
#     def construct(self, x):
#         out = x * nn.ReLU6()(x + 3) / 6
#         return out


# class HSigmoid(nn.Cell):
#     def construct(self, x):
#         out = nn.ReLU6()(x + 3) / 6
#         return out

class SE(nn.Cell):
    def __init__(self, num_feat, reduction=4):
        super().__init__()
        self.fc = nn.SequentialCell(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // reduction, 1),
            nn.BatchNorm2d(num_feat // reduction),
            nn.ReLU(),
            nn.Conv2d(num_feat // reduction, num_feat, 1),
            nn.BatchNorm2d(num_feat),
            nn.HSigmoid(),
        )

    def construct(self, x):
        y = self.fc(x)
        return x * y


class Block(nn.Cell):
    '''expand + depthwise + pointwise'''

    def __init__(
        self,
        kernel_size,
        in_size,
        expand_size,
        out_size,
        nolinear,
        stride,
        semodule=False,
    ):
        super(Block, self).__init__()
        self.stride = stride

        self.se = SE(out_size) if semodule else None

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(
            expand_size,
            expand_size,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode='pad',
            padding=kernel_size // 2,
            group=expand_size,
        )
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(
            expand_size, out_size, kernel_size=1, stride=1, padding=0
        )
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.SequentialCell()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.SequentialCell(
                nn.Conv2d(in_size, out_size, kernel_size=1,
                          stride=1, padding=0),
                nn.BatchNorm2d(out_size),
            )

    def construct(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se is not None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV3_Large(nn.Cell):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = nn.HSwish()

        self.bneck = nn.SequentialCell(
            Block(3, 16, 16, 16, nn.ReLU(), 1),
            Block(3, 16, 64, 24, nn.ReLU(), 2),
            Block(3, 24, 72, 24, nn.ReLU(), 1),
            Block(5, 24, 72, 40, nn.ReLU(), 2, True),
            Block(5, 40, 120, 40, nn.ReLU(), 1, True),
            Block(5, 40, 120, 40, nn.ReLU(), 1, True),
            Block(3, 40, 240, 80, nn.HSwish(), 2),
            Block(3, 80, 200, 80, nn.HSwish(), 1),
            Block(3, 80, 184, 80, nn.HSwish(), 1),
            Block(3, 80, 184, 80, nn.HSwish(), 1),
            Block(3, 80, 480, 112, nn.HSwish(), 1, True),
            Block(3, 112, 672, 112, nn.HSwish(), 1, True),
            Block(5, 112, 672, 160, nn.HSwish(), 1, True),
            Block(5, 160, 672, 160, nn.HSwish(), 2, True),
            Block(5, 160, 960, 160, nn.HSwish(), 1, True),
        )

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = nn.HSwish()
        self.linear3 = nn.Dense(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = nn.HSwish()
        self.linear4 = nn.Dense(1280, num_classes)

        self.avg_pool2d = nn.AvgPool2d(7)

    def construct(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.avg_pool2d(out)
        out = out.view(out.shape[0], -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out


class MobileNetV3_Small(nn.Cell):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = nn.HSwish()

        self.bneck = nn.SequentialCell(
            Block(3, 16, 16, 16, nn.ReLU(), 2, True),
            Block(3, 16, 72, 24, nn.ReLU(), 2),
            Block(3, 24, 88, 24, nn.ReLU(), 1),
            Block(5, 24, 96, 40, nn.HSwish(), 2, True),
            Block(5, 40, 240, 40, nn.HSwish(), 1, True),
            Block(5, 40, 240, 40, nn.HSwish(), 1, True),
            Block(5, 40, 120, 48, nn.HSwish(), 1, True),
            Block(5, 48, 144, 48, nn.HSwish(), 1, True),
            Block(5, 48, 288, 96, nn.HSwish(), 2, True),
            Block(5, 96, 576, 96, nn.HSwish(), 1, True),
            Block(5, 96, 576, 96, nn.HSwish(), 1, True),
        )

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = nn.HSwish()
        self.linear3 = nn.Dense(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = nn.HSwish()
        self.linear4 = nn.Dense(1280, num_classes)

        self.avg_pool2d = nn.AvgPool2d(7)

    def construct(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.avg_pool2d(out)
        out = out.view(out.shape[0], -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out
        
        
if __name__ == "__main__":
    net = MobileNetV3_Large()
    register_hook(net, acc_cmp_dump)
    set_dump_switch("ON")
    grad_net = ms.grad(net)
    output = grad_net(ms.Tensor(np.random.random([4, 3, 224, 224]).astype(np.float32)))

    print(output)
