from mindspore import ops
from mindspore import Tensor
import numpy as np

input_size = 10
hidden_size = 2
num_layers = 1
seq_len = 5
batch_size = 2

# ascend device not support.
net = ops.LSTM(input_size, hidden_size, num_layers, True, False, 0.0)
input_tensor = Tensor(np.ones([seq_len, batch_size, input_size]).astype(np.float32))
h0 = Tensor(np.ones([num_layers, batch_size, hidden_size]).astype(np.float32))
c0 = Tensor(np.ones([num_layers, batch_size, hidden_size]).astype(np.float32))
w = Tensor(np.ones([112, 1, 1]).astype(np.float32))
output, hn, cn, _, _ = net(input_tensor, h0, c0, w)
print(output)










