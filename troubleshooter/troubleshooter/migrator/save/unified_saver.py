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

from __future__ import absolute_import

import mindspore as ms
import torch
from troubleshooter.migrator.save.mindspore_saver import SaveTensorMs, _SaveTensorMs


class SaveTensorMsPt(SaveTensorMs):
    """
    The SaveNet class is used to build a unified data storage interface that supports PyTorch and MindSpore
    PYNATIVE_MODE as well as GRAPH_MODE, but currently does not support MindSpore GRAPH_MODE.

    Inputs:
        file (str): The name of the file to be stored.
        data (Union(mindspore.Tensor, torch.tensor)): Supports data types of Tensor for both MindSpore and PyTorch.

    Outputs:
        The output storage name is 'id_name.npy'.
    """

    def numpy(self, data):
        if isinstance(data, ms.Tensor):
            return data.asnumpy()
        elif torch.is_tensor(data):
            return data.cpu().detach().numpy()
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, " \
                            f"but got {type(data)}")


save = SaveTensorMsPt()


class _SaveTensorMsPt(_SaveTensorMs):
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

    def numpy(self, data):
        if isinstance(data, ms.Tensor):
            return data.asnumpy()
        elif torch.is_tensor(data):
            return data.cpu().detach().numpy()
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, " \
                            f"but got {type(data)}")


_save = _SaveTensorMsPt()
