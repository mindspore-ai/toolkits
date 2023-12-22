# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Define the constants for Network Validation."""
import re

import numpy as np


GE_GRAPH_TYPE = "ge"
OM_GRAPH_TYPE = "om"
GEIR_GRAPH_TYPE = "geir"
ME_GRAPH_TYPE = "me"

SUPPORTED_GRAPH_TYPE = [ME_GRAPH_TYPE]

VER_X = 'x'
VER_X_ALIASES = ('x', '')


ME_VER_11X = '1.1.x'
ME_VER_11X_ALIASES = ('1.1', '1.1.0', '1.1.x')
ME_VER_11X_OPT = '1.1.x-opt'
ME_VER_11X_OPT_ALIASES = ('1.1-opt', '1.1.0-opt', '1.1.x-opt')
ME_VER_11X_GPU = '1.1.x-gpu'
ME_VER_11X_GPU_ALIASES = ('1.1-gpu', '1.1.0-gpu', '1.1.x-gpu')
ME_VER_11X_GPU_OPT = '1.1.x-gpu-opt'
ME_VER_11X_GPU_OPT_ALIASES = ('1.1-gpu-opt', '1.1.0-gpu-opt', '1.1.x-gpu-opt')

ME_VER_20X = '2.0.x'
ME_VER_20X_ALIASES = ('2', '2.0', '2.0.0', '2.0.x')
ME_VER_20X_OPT = '2.0.x-opt'
ME_VER_20X_OPT_ALIASES = ('2-opt', '2.0-opt', '2.0.0-opt', '2.0.x-opt')
ME_VER_20X_GPU = '2.0.x-gpu'
ME_VER_20X_GPU_ALIASES = ('2-gpu', '2.0-gpu', '2.0.0-gpu', '2.0.x-gpu')
ME_VER_20X_GPU_OPT = '2.0.x-gpu-opt'
ME_VER_20X_GPU_OPT_ALIASES = ('2-gpu-opt', '2.0-gpu-opt', '2.0.0-gpu-opt', '2.0.x-gpu-opt')


ME_TYPE_VER_X = ME_GRAPH_TYPE + '-' + VER_X

ME_TYPE_VER_11X = ME_GRAPH_TYPE + '-' + ME_VER_11X
ME_TYPE_VER_11X_OPT = ME_GRAPH_TYPE + '-' + ME_VER_11X_OPT
ME_TYPE_VER_11X_GPU = ME_GRAPH_TYPE + '-' + ME_VER_11X_GPU
ME_TYPE_VER_11X_GPU_OPT = ME_GRAPH_TYPE + '-' + ME_VER_11X_GPU_OPT

ME_TYPE_VER_20X = ME_GRAPH_TYPE + '-' + ME_VER_20X
ME_TYPE_VER_20X_OPT = ME_GRAPH_TYPE + '-' + ME_VER_20X_OPT
ME_TYPE_VER_20X_GPU = ME_GRAPH_TYPE + '-' + ME_VER_20X_GPU
ME_TYPE_VER_20X_GPU_OPT = ME_GRAPH_TYPE + '-' + ME_VER_20X_GPU_OPT

ME_TYPE_VER_TRACE_CODE = ME_GRAPH_TYPE + "-" + "trace_code"
ME_TYPE_VER_TRACE_CODE_OPT = ME_TYPE_VER_TRACE_CODE + "-opt"


DATA_TYPE_PATTERN = re.compile(r"(bool|int\d+|uint\d+|float\d+)")

DATA_TYPE = {
    "bool": np.bool_,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.uint64,
}
