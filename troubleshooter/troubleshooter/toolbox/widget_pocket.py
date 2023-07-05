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
from troubleshooter.common.util import validate_and_normalize_path, print_to_file
from troubleshooter import FRAMEWORK_TYPE

__all__ = [
    "fix_random",
    "get_pt_grads",
    "load_ms_weight2net",
    "object_load",
    "object_dump",
    "save_net_and_weight_params"
]

if "torch" in FRAMEWORK_TYPE:
    import torch

if "mindspore" in FRAMEWORK_TYPE:
    import mindspore

def fix_random(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    if "torch" in FRAMEWORK_TYPE:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.manual_seed(seed)
    if "mindspore" in FRAMEWORK_TYPE:
        mindspore.set_seed(seed)

def save_net_and_weight_params(model, path=os.getcwd()):
    if not os.path.exists(path):
        os.makedirs(path)

    if "torch" in FRAMEWORK_TYPE and isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), os.path.join(path, "torch.pth"))
        from troubleshooter.migrator import get_weight_map
        get_weight_map(model, weight_map_save_path=os.path.join(path, "torch_net_map.json"))
        print_to_file(model, os.path.join(path, "torch_model_architecture.txt"))
        return

    if "mindspore" in FRAMEWORK_TYPE and isinstance(model, mindspore.nn.Cell):
        mindspore.save_checkpoint(model, os.path.join(path, "mindspore.ckpt"))
        print_to_file(model, os.path.join(path, "mindspore_model_architecture.txt"))
        return

    raise ValueError("For the 'save_net_and_weight_params', the type of the 'model' parameter must be" \
                     "'MindSpore.nn.Cell' or 'torch.nn.Module'.")

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

