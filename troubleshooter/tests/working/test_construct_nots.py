#!/usr/bin/env python3
# coding=utf-8

"""
This is a python code template
"""

from mindspore.nn import Cell

import troubleshooter as ts


class Net(Cell):
    def construct(x):
        return x


class Net_LessInput(Cell):
    def construct(self, x, y):
        return x + y


class Net_MoreInput(Cell):
    def construct(self, x):
        return x


def less_input_case():
    net = Net_LessInput()
    out = net(1)
    print(out)


def more_input_case():
    net = Net_MoreInput()
    out = net(1, 2)
    print(out)


def main():
    net = Net()
    out = net(2)
    print("out=", out)


if __name__ == '__main__':
    # main()
    # less_input_case()
    more_input_case()
