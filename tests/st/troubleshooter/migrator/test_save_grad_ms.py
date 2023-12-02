import troubleshooter as ts
import time
import numpy as np
import mindspore as ms
import pytest
from mindspore.common.initializer import initializer, Zero

from tempfile import TemporaryDirectory
from pathlib import Path
from troubleshooter.common.util import find_file


class NetWithSaveGrad(ms.nn.Cell):
    def __init__(self, path):
        super(NetWithSaveGrad, self).__init__()
        self.dense = ms.nn.Dense(3, 2)
        self.apply(self._init_weights)
        self.path = path

    def _init_weights(self, cell):
        if isinstance(cell, ms.nn.Dense):
            cell.weight.set_data(initializer(Zero(), cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer(Zero(), cell.bias.shape, cell.bias.dtype))

    def construct(self, x):
        x = self.dense(x)
        x = ts.save_grad(self.path, x)
        return x

class NetWithParameterSaveGrad(ms.nn.Cell):
    def __init__(self):
        super(NetWithParameterSaveGrad, self).__init__()
        self.dense = ms.nn.Dense(3, 2)
        self.dense_grad = ms.Parameter(ms.ops.zeros((2), ms.float32), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, cell):
        if isinstance(cell, ms.nn.Dense):
            cell.weight.set_data(initializer(Zero(), cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer(Zero(), cell.bias.shape, cell.bias.dtype))

    def construct(self, x):
        x = self.dense(x)
        x = x + self.dense_grad
        return x

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ms_save_grad_should_equal_to_parameter_save(mode):
    ms.set_context(mode=mode)

    temp_dir = TemporaryDirectory()
    path = Path(temp_dir.name)
    file_name = path / "dense"
    data = np.array([0.2, 0.5, 0.2], dtype=np.float32)
    label = np.array([1, 0], dtype=np.float32)

    net1 = NetWithSaveGrad(str(file_name))
    net2 = NetWithParameterSaveGrad()
    loss_fn = ms.nn.CrossEntropyLoss()

    def forward_fn1(data, label):
        logits = ms.ops.squeeze(net1(data))
        loss = loss_fn(logits, label)
        return loss, logits

    def forward_fn2(data, label):
        logits = ms.ops.squeeze(net2(data))
        loss = loss_fn(logits, label)
        return loss, logits

    grad_fn1 = ms.grad(forward_fn1, grad_position=None, weights=net1.trainable_params(), has_aux=True, return_ids=True)
    grad_fn2 = ms.grad(forward_fn2, grad_position=None, weights=net2.trainable_params(), has_aux=True, return_ids=True)
    grads, _ = grad_fn1(ms.ops.unsqueeze(ms.Tensor(data), dim=0), ms.Tensor(label))
    grads, _ = grad_fn2(ms.ops.unsqueeze(ms.Tensor(data), dim=0), ms.Tensor(label))
    expected = None
    for name, grad in grads:
        if name == 'dense_grad':
            expected = grad.asnumpy()
    time.sleep(0.1)
    file_list = find_file(path)
    assert np.allclose(expected, np.load(str(path / file_list[0])))
