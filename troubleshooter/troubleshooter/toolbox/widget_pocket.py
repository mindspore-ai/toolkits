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
import random
import os
import numpy as np

FW_PT=True
FW_MS=True


try:
    import torch
except ModuleNotFoundError as e:
    e_msg = e.msg
    no_module_msg = "No module named 'torch'"
    if e_msg != no_module_msg:
        raise e
    else:
        FW_PT = False


try:
    import mindspore
except ModuleNotFoundError as e:
    e_msg = e.msg
    no_module_msg = "No module named 'mindspore'"
    if e_msg != no_module_msg:
        raise e
    else:
        FW_MS = False


def fix_random(seed):
        random.seed(seed)
        os.environ["PYTHONSEED"] = str(seed)
        np.random.seed(seed)
        if FW_PT:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.manual_seed(seed)
        if FW_MS:
            mindspore.set_seed(seed)


def get_pt_grads(model):
    grads_dict = {}
    i=0
    for name ,params in model.named_parmeters():
        name = name + '-' + str(i)
        grads_dict.update({name:params.grad})
        i += 1
    return grads_dict


def load_ms_weight2net(model,file):
    param = mindspore.load_checkpoint(file)
    mindspore.load_param_into_net(model, param)

def _object_load(file):
    import pickle
    with open(file,'rb') as f:
        obj = pickle.load(f)
    return obj

def _object_dump(obj, file):
    import pickle
    with open(file,'wb') as f:
        pickle.dump(obj,f)

