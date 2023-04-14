#!/usr/bin/env python3
# coding=utf-8

"""
This is a python code template
"""

import os
from mindspore import Tensor, nn, ops, context
import mindspore
import numpy as np
from tests.util import delete_file, file_and_key_match
import troubleshooter as ts


def test_ops_conv2d():
    delete_file("/tmp/")
    with ts.proposal(write_file_path="/tmp/"):
        net = nn.Conv2d(120, 240, 4, has_bias=False, weight_init='normal')
        x = Tensor(np.ones([120, 1024, 640]), mindspore.float32)
        output = net(x)
        print(output)

    assert file_and_key_match("/tmp/", "operator_id_1")


def test_ops_loss_function():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        logits = Tensor(np.array([[-0.8, 1.2], [-0.1, -0.4]]).astype(np.float32))
        labels = Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]).astype(np.float32))
        # loss = nn.BCEWithLogitsLoss()
        loss = nn.MSELoss()
        output = loss(logits, labels)
        print(output)

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "operator_id_9")


def test_ops_mul():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        y = np.array([4.0, 5.0, 6.0])
        output = x * y
        print(output)

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "operator_id_18")


def test_ops_padding_4d():
    delete_file("/tmp/")
    with ts.proposal(write_file_path="/tmp/"):
        paddings = ((1, 1), (2, 2))
        net = nn.Pad(paddings, mode="SYMMETRIC")
        x = Tensor(np.array([[[[1, 2, 3], [4, 5, 6]]]]), mindspore.float32)
        print("x=", x.shape)
        y = net(x)
        print(y.shape)

    assert file_and_key_match("/tmp/", "operator_id_3")


def test_ops_padding_5d():
    delete_file("/tmp/")
    with ts.proposal(write_file_path="/tmp/"):
        paddings = ((0, 0), (0, 0), (0, 0), (1, 1), (2, 2))
        net = nn.Pad(paddings, mode="SYMMETRIC")
        x = Tensor(np.array([[[[[1, 2, 3], [4, 5, 6]]]]]), mindspore.float32)
        print("x=", x.shape)
        y = net(x)
        print(y.shape)
    assert file_and_key_match("/tmp/", "operator_id_5")


def test_ops_padding_negative_index():
    delete_file("/tmp/")
    with ts.proposal(write_file_path="/tmp/"):
        paddings = ((-1, 1), (2, 2))
        net = nn.Pad(paddings, mode="SYMMETRIC")
        x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), mindspore.float32)
        print("x=", x.shape)
        y = net(x)
        print(y.shape)
    assert file_and_key_match("/tmp/", "operator_id_2")


"""
def test_ops_pow():
    # device_type = "Ascend"
    # context.set_context(device_target=device_type)
    @ts.proposal(write_file_path="/tmp/")
    def main():
        input_x = Tensor(np.array([[1.0], [2.0], [-4.0]]), mindspore.float32)
        input_y = 0.5
        _pow = ops.Pow()
        out = _pow(input_x, input_y)
        print(out)
        print(out.dtype)

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "operator_id_18")
"""


def test_ops_reducemean():
    context.set_context(mode=context.GRAPH_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.reduce_mean = ops.ReduceMean(keep_dims=True)

        def construct(self, x):
            out = self.reduce_mean(x, axis=(2, 3))
            return out

    delete_file("/tmp/")
    with ts.proposal(write_file_path="/tmp/"):
        net = Net()
        x = Tensor(np.ones((3, 4, 5, 6), dtype=np.float32), mindspore.float32)
        out = net(x)
        print('out', out.shape)
    assert file_and_key_match("/tmp/", "operator_id_13")


def test_ops_scalar_add():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        x = False
        y = True
        output = ops.scalar_add(x, y)
        print('output', output)

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "operator_id_16")


def test_ops_sub():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        x = Tensor(np.array([2, 3]), mindspore.int32)
        y = Tensor(np.array([4, 5, 6]), mindspore.int32)
        output = x - y
        print(output)

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "operator_id_9")
