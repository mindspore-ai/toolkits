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
import os
import random

import numpy as np

from troubleshooter import FRAMEWORK_TYPE
from troubleshooter.common.util import validate_and_normalize_path

__all__ = [
    "fix_random",
    "get_pt_grads",
    "load_ms_weight2net",
    "object_load",
    "object_dump"
]

if "torch" in FRAMEWORK_TYPE:
    import torch

if "mindspore" in FRAMEWORK_TYPE:
    import mindspore

def fix_random(seed=16):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    if "torch" in FRAMEWORK_TYPE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
    if "mindspore" in FRAMEWORK_TYPE:
        mindspore.set_seed(seed)

def get_pt_grads(model):
    grads_dict = {}
    for idx, (name, params) in enumerate(model.named_parameters()):
        name = f"{name}_{idx}"
        grads_dict.update({name: params.grad})
    return grads_dict


def load_ms_weight2net(model,file):
    file = validate_and_normalize_path(file)
    param = mindspore.load_checkpoint(file)
    mindspore.load_param_into_net(model, param)

def _object_p_load(file):
    import pickle
    file = validate_and_normalize_path(file)
    with open(file,'rb') as f:
        obj = pickle.load(f)
    return obj

def _object_p_dump(obj, file):
    import pickle
    file = validate_and_normalize_path(file)
    with open(file,'wb') as f:
        pickle.dump(obj,f)

def object_load(file):
    import json
    file = validate_and_normalize_path(file)
    with open(file,'r') as f:
        obj = json.load(f)
    return obj

def object_dump(obj, file):
    import json
    file = validate_and_normalize_path(file)
    with open(file,'w') as f:
        json.dump(obj,f)

