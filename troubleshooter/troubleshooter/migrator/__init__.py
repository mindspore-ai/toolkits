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
"""troubleshooter.common"""
from __future__ import absolute_import

from troubleshooter import FRAMEWORK_TYPE
from troubleshooter.migrator import api_dump, diff_handler
from troubleshooter.migrator.api_dump import *
from troubleshooter.migrator.diff_handler import *
from troubleshooter.migrator.save import save

__all__ = ["save",]
__all__.extend(diff_handler.__all__)
__all__.extend(api_dump.__all__)

if {"torch", "mindspore"}.issubset(FRAMEWORK_TYPE):
    from troubleshooter.migrator import net_diff_finder, weight_migrator
    from troubleshooter.migrator.net_diff_finder import *
    from troubleshooter.migrator.weight_migrator import *
    __all__.extend(net_diff_finder.__all__)
    __all__.extend(weight_migrator.__all__)
