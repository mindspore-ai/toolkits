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
"""Commandline module helper."""
import sys
import os


if sys.argv and sys.argv[0] != 'mindcompare':
    # for direct usage from source code pack
    script_dir = os.path.dirname(os.path.realpath(__file__))
    mi_dir = os.path.realpath(os.path.join(script_dir, '../../'))
    sys.path.insert(0, mi_dir)


def print_info():
    if sys.argv and sys.argv[0] == 'mindcompare':
        print("run from mindcompare command")
    else:
        print("run from source code pack")
