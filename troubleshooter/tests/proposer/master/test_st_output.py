#!/usr/bin/env python3
# coding=utf-8

"""
This is a python code template
"""

import os, re
from mindspore import Tensor, nn, ops, context
import mindspore
import numpy as np
from tests.util import delete_file, file_and_key_match
import troubleshooter as ts


def test_ops_reducemean():
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

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
    id = file_and_key_match("/tmp/", "operator_id_13")
    mode = file_and_key_match("/tmp/", r"Graph Mode")
    device = file_and_key_match("/tmp/", r"CPU")
    error_code_case = file_and_key_match("/tmp/", "out = reduce_mean(x, axis=(2, 3))")
    correct_code_case = file_and_key_match("/tmp/", "out = reduce_mean(x, (2, 3))")
    suggestion = file_and_key_match("/tmp/", "查询算子API接口说明，确认输入参数类型，修改至符合要求")
    case = file_and_key_match("/tmp/", "https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=182167")
    assert id and mode and device and error_code_case and correct_code_case and case and suggestion


def test_print_clear_stack_false(capsys):
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    with ts.proposal():
        net = nn.Conv2d(120, 240, 4, has_bias=False, weight_init='normal')
        x = Tensor(np.ones([120, 1024, 640]), mindspore.float32)
        output = net(x).shape
        print(output)
    result = capsys.readouterr().err
    assert re.search("cell.py.*in __call__", result) and re.search("primitive.py.*__call__", result) and \
           re.search("cell.py.* _run_construct", result)


def test_print_clear_stack_true(capsys):
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    with ts.proposal(print_clear_stack=True):
        net = nn.Conv2d(120, 240, 4, has_bias=False, weight_init='normal')
        x = Tensor(np.ones([120, 1024, 640]), mindspore.float32)
        output = net(x).shape
        print(output)
    result = capsys.readouterr().err
    assert re.search("cell.py.*in __call__", result) is None and \
           re.search("primitive.py.*__call__", result) is None and \
           re.search("cell.py.* _run_construct", result) is None