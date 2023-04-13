# Copyright 2022 Tiger Miao and collaborators.
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
"""util functions"""
import re
import os
import stat
from troubleshooter.common.util import validate_and_normalize_path


def delete_file(path, file_name="mindspore_failure_analysis_report.log"):
    path = validate_and_normalize_path(path)
    file = os.path.join(path, file_name)
    if os.path.exists(file):
        # 删除文件，可使用以下两种方法。
        os.remove(file)


def file_and_key_match(path, key, file_name="mindspore_failure_analysis_report.log"):
    path = validate_and_normalize_path(path)
    file = os.path.join(path, file_name)
    with open(file, "r") as f:
        result = f.read()
    if result.find(key) == -1:
        result = False
    else:
        result = True
    return result

