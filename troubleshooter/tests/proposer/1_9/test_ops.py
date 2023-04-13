#!/usr/bin/env python3
# coding=utf-8

import numpy as np
import mindspore
from tests.util import delete_file, file_and_key_match
import troubleshooter as ts


def test_ops_view():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        data = mindspore.Tensor(np.random.randint(100, size=(20, 4)), mindspore.float32)
        data = data.view((-1, 3))
        print(data.shape)

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "operator_id_11")
