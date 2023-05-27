#!/usr/bin/env python3
# coding=utf-8

"""
This is a python code template
"""

import mindspore as ms
import mindspore.nn as nn
import numpy as np
import mindspore.ops as ops
from mindspore import Tensor
from troubleshooter import proposal


class IfInWhileNet(nn.Cell):

    def __init__(self):
        super().__init__()
        self.nums = [1, 2, 3]

    def construct(self, x, y, i):
        j = 0
        out = x

        # 构造条件表达式为变量条件的while语句
        while i < 3:
            if x + i < y:
                out = out + x
            else:
                out = out + y
            out = out + self.nums[j]
            i = i + 1
            # 在条件表达式为变量条件的while语句循环体内构造标量计算
            j = j + 1

        return out


with proposal():
    forward_net = IfInWhileNet()
    i = Tensor(np.array(0), dtype=ms.int32)
    x = Tensor(np.array(0), dtype=ms.int32)
    y = Tensor(np.array(1), dtype=ms.int32)

    output = forward_net(x, y, i)
    print(output)
