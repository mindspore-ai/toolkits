# Copyright 2022-2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import mindspore as ms
import numpy as np
import os
import torch
from collections import OrderedDict


class SaveTensorMsPt(ms.nn.Cell):
    """
    The SaveNet class is used to build a unified data storage interface that supports PyTorch and MindSpore
    PYNATIVE_MODE as well as GRAPH_MODE, but currently does not support MindSpore GRAPH_MODE.

    Inputs:
        file (str): The name of the file to be stored.
        data (Union(mindspore.Tensor, torch.tensor)): Supports data types of Tensor for both MindSpore and PyTorch.

    Outputs:
        The output storage name is 'id_name.npy'.
    """

    def __init__(self):
        super(SaveTensorMsPt, self).__init__()
        self.cnt = ms.Parameter(ms.Tensor(0, ms.int32), name="cnt", requires_grad=False)
        self.sep = os.sep

    def numpy(self, data):
        if isinstance(data, ms.Tensor):
            return data.asnumpy()
        elif torch.is_tensor(data):
            return data.cpu().detach().numpy()
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, " \
                            f"but got {type(data)}")

    def handle_path(self, file):
        if file[-1] == self.sep:
            raise ValueError(f"For 'ts.save', the type of argument 'file' must be a valid filename, but got {file}")
        name = ''
        for c in file:
            if c == self.sep:
                name = ''
            else:
                name += c
        path = ''
        for i in range(len(file) - len(name)):
            path += file[i]
        return path, name

    def construct(self, file, data):
        if file:
            path, name = self.handle_path(file)
        else:
            path, name = '', 'tensor_' + str(tuple(data.shape))
        np.save(f"{path}{int(self.cnt)}_{name}", self.numpy(data))
        self.cnt += 1
        return


save = SaveTensorMsPt()


class _SaveNet(ms.nn.Cell):
    """
    The SaveNet class is used to build a unified data storage interface that supports PyTorch and MindSpore
    PYNATIVE_MODE as well as GRAPH_MODE, but currently does not support MindSpore GRAPH_MODE.

    Inputs:
        file (str): The name of the file to be stored.
        data (Union(Tensor, list[Tensor], Tuple[Tensor], dict[str, Tensor])): Supports data types of Tensor,
          list[Tensor], tuple(Tensor), and dict[str, Tensor] for both MindSpore and PyTorch. When the input is
          a list or tuple of Tensor, the file name will be numbered according to the index of the Tensor.
          When the input is a dictionary of Tensor, the corresponding key will be added to the file name.
        auto_id (bool): Whether to enable automatic numbering. If set to True, an incremental number will be
          added before the saved file name. If set to False, no numbering will be added to the file name.
        suffix (str): The suffix of the saved file name.

    Outputs:
        The output storage name is '{id}_name_{idx/key}_{suffix}.npy'.
    """

    def __init__(self):
        super(_SaveNet, self).__init__()
        self.cnt = ms.Parameter(ms.Tensor(0, ms.int32), name="cnt", requires_grad=False)
        self.sep = os.sep

    def numpy(self, data):
        if isinstance(data, ms.Tensor):
            return data.asnumpy()
        elif torch.is_tensor(data):
            return data.cpu().detach().numpy()
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, " \
                            f"but got {type(data)}")

    def handle_path(self, file):
        if file[-1] == self.sep:
            raise ValueError(f"For 'ts.save', the type of argument 'file' must be a valid filename, but got {file}")
        name = ''
        for c in file:
            if c == self.sep:
                name = ''
            else:
                name += c
        path = ''
        for i in range(len(file) - len(name)):
            path += file[i]
        return path, name

    def construct(self, file, data, auto_id, suffix):
        path, name = self.handle_path(file)
        if isinstance(data, (list, tuple)):
            for idx, val in enumerate(data):
                if auto_id:
                    np.save(f"{path}{int(self.cnt)}_{name}_{idx}_{suffix}" if suffix else
                            f"{path}{int(self.cnt)}_{name}_{idx}", self.numpy(val))
                else:
                    np.save(f"{file}_{idx}_{suffix}" if suffix else
                            f"{file}_{idx}", self.numpy(val))
        elif isinstance(data, (dict, OrderedDict)):
            for key, val in data.items():
                if auto_id:
                    np.save(f"{path}{int(self.cnt)}_{name}_{key}_{suffix}" if suffix else
                            f"{path}{int(self.cnt)}_{name}_{key}", self.numpy(val))
                else:
                    np.save(f"{file}_{key}_{suffix}" if suffix else
                            f"{file}_{key}", self.numpy(val))
        else:
            if auto_id:
                np.save(f"{path}{int(self.cnt)}_{name}_{suffix}" if suffix else
                        f"{path}{int(self.cnt)}_{name}", self.numpy(data))
            else:
                np.save(f"{file}_{suffix}" if suffix else file,
                        self.numpy(data))
        if auto_id:
            self.cnt += 1
