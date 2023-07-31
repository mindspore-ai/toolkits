import mindspore as ms
import numpy as np
from mindspore import Tensor, ops, nn
from troubleshooter.migrator import api_dump_init, api_dump_start

ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = ops.norm(x)
        x = self.relu(x)
        x = ops.norm(x)
        x = self.relu(x)
        x = ops.clip(x, Tensor(0.2, ms.float32), Tensor(0.5, ms.float32))
        x = x.abs()
        x = ops.ravel(x)
        return x


if __name__ == "__main__":
    net = Net()
    api_dump_init(net)
    api_dump_start()
    grad_net = ms.grad(net, None, net.trainable_params())
    output = grad_net(ms.Tensor(np.random.random([1, 1, 2, 2]).astype(np.float32)))
    print(output)
