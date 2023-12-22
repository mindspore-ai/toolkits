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
"""Computational graph of debugger trace code graph."""
import mindinsightx.compare.proto.me.debugger_graph_pb2 as debugger_graph_pb2
from .me11x_graph import Me11xGraph


class MeTraceCodeGraph(Me11xGraph):
    def _get_model_proto(self):
        return debugger_graph_pb2.ModelProto()
