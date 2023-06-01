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


weight_name_map = {"torch.nn.BatchNorm1d":{'weight': 'gamma', 'bias': 'beta', 'running_mean': 'moving_mean' , 'running_var': 'moving_variance'},
    "torch.nn.BatchNorm2d":{'weight':'gamma', 'bias':'beta', 'running_mean':'moving_mean' ,'running_var':'moving_variance'},
    "torch.nn.BatchNorm3d":{'weight':'bn2d.gamma', 'bias':'bn2d.beta', 'running_mean':'bn2d.moving_mean' ,'running_var':'bn2d.moving_variance'},
    "torch.nn.Embedding":{'weight':'embedding_table'},
    "torch.nn.GroupNorm":{'weight':'gamma', 'bias': 'beta'},
    "torch.nn.LayerNorm":{'weight': 'gamma', 'bias': 'beta'},
    "torch.nn.PReLU":{'weight':'a'},
                   }

weight_value_map = {"torch.nn.Conv1d":{'weight': 'troubleshooter.migrator.mapping_relation.weight_adapter.trans_conv1d_weight_value'}
                    }