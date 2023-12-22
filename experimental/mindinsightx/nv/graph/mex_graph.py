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
"""Computational graph of MindSpore unknown version."""
from mindinsightx.nv import constants
from mindinsightx.nv.graph.me_graph import MeGraph


class MeXGraph(MeGraph):
    """Computational graph of MindSpore unknown version."""

    def __init__(self, framework_version):
        super().__init__()
        self.FRAMEWORK_VERSION = framework_version

    @property
    def framework_version(self):
        return self.FRAMEWORK_VERSION

    def keep_shape(self, node_type):
        return True

    def keep_shape_remove_batch(self, node_type):
        return False

    def get_node_type_map(self, type_ver_code):
        return None
