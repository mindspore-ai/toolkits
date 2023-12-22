import mindspore.ops
import mindspore.nn as nn
import mindspore.experimental.optim as optim
import mindspore.ops as ops


class TestLeNet(nn.Cell):
    """TestLeNet network."""
    def __init__(self):
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Dense(in_channels=16 * 5 * 5, out_channels=120)
        self.fc2 = nn.Dense(in_channels=120, out_channels=84)
        self.fc3 = nn.Dense(in_channels=84, out_channels=10)

    def construct(self, input_x):
        """Callback method."""
        out = ops.relu(input=self.conv1(input_x))
        out = ops.MaxPool(ksize=2, strides=2, padding='valid', input=out)
        out = ops.relu(input=self.conv2(out))
        out = ops.MaxPool(ksize=2, strides=2, padding='valid', input=out)
        out = out.view(out.shape(0), -1)
        out = ops.relu(input=self.fc1(out))
        out = ops.relu(input=self.fc2(out))
        out = self.fc3(out)
        return out