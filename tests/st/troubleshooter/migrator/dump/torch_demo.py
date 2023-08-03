from troubleshooter.migrator import api_dump_init, api_dump_start
import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
import torch.optim as optim



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.linalg.norm(x)
        x = self.relu(x)
        x = torch.linalg.norm(x)
        x = self.relu(x)
        x = torch.clip(x, 0.2, 0.5)
        x = x.abs()
        x = torch.ravel(x)
        return x


if __name__ == "__main__":
    net = Net()
    api_dump_init(net, "torch_dump")
    api_dump_start()
    criterion = nn.MSELoss()   # 均方损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    inputs = torch.randn(2, 1, 2, 2)
    label = torch.randn(1)
    out = net(inputs)
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()
