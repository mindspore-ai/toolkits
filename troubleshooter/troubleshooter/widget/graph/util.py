# Copyright 2023 Huawei Technologies Co., Ltd
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
import enum
import os.path


class BaseEnum(enum.Enum):

    @classmethod
    def list_members(cls):
        """List all members."""
        return [member.value for member in cls]


class NodeTypeEnum(BaseEnum):
    """Node type enum. The following types are new to our custom."""
    NAME_SCOPE = 'name_scope'
    AGGREGATION_SCOPE = 'aggregation_scope'
    PARAMETER = 'Parameter'
    CONST = 'Const'
    LOAD = 'Load'
    MAKETUPLE = 'MakeTuple'
    TUPLE_GET_ITEM = 'TupleGetItem'
    UPDATE_STATE = 'UpdateState'


class PluginNameEnum(BaseEnum):
    """Plugin Name Enum."""
    IMAGE = 'image'
    SCALAR = 'scalar'
    GRAPH = 'graph'
    OPTIMIZED_GRAPH = 'optimized_graph'
    HISTOGRAM = 'histogram'
    TENSOR = 'tensor'
    LANDSCAPE = 'loss_landscape'


class EdgeTypeEnum(BaseEnum):
    """Node edge type enum."""
    CONTROL = 'control'
    DATA = 'data'


def check_invalid_character(string):
    """Check for invalid characters. These characters will cause frontend crash."""
    invalid_char = {'>', '<', '"'}
    result = set(string).intersection(invalid_char)
    if result:
        raise Exception(f"There are some invalid characters in graph node, invalid string: {string}, "
                        f"unexpected characters: {result}")


def check_invalid_pb_file(file):
    if not file:
        raise Exception('Please pass in the source file')

    if not isinstance(file, str):
        raise Exception(f'Source file arg type must be str, but got {type(file)}')

    if file.split('.')[-1] != 'pb':
        raise Exception(f'Source file is not pb file, file : {file}')

    if not os.path.exists(file):
        raise Exception(f'Source file not found, please check the valid path file. file : {file}')

    return
