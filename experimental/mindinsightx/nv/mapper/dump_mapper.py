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
"""Dump file mapper."""
import copy

from mindinsightx.nv.constants import DATA_TYPE_PATTERN, DATA_TYPE


class DumpRecord:
    """Dump record that stores dump file information."""
    def __init__(self, real_path, filename, index, shape=None, dtype=None, data_format=None, timestamp=None):
        self.real_path = real_path
        self.filename = filename
        self.index = index
        self.shape = shape
        self.dtype = dtype
        self.data_format = data_format
        self.timestamp = timestamp

    def load(self):
        """Load the dump file content."""
        raise NotImplementedError


class DumpMapper:
    """Node to dump file mapper."""

    def __init__(self, graph, dump_dir):
        """
        Initialize the mapping.

        Args:
            dump_dir (str): The dump directory.
            graph (Graph): The graph instance.
        """
        self._dump_dir = dump_dir
        self._graph = graph
        self._input_map = None
        self._output_map = None

    @property
    def dump_dir(self):
        """The dump directory."""
        return self._dump_dir

    @property
    def input_map(self):
        """
        The node id-output dump files mapping.

        Returns:
            dict[int, list[DumpRecord]]
        """
        return self._input_map

    @property
    def output_map(self):
        """
        The node id-output dump files mapping.

        Returns:
            dict[int, list[DumpRecord]]
        """
        return self._output_map

    def process(self, map_input=True, map_output=True):
        """Build up the mapping."""
        raise NotImplementedError

    def _copy_rec_from_input_node(self, node_types, mapping):
        """Copy the input node's dump records if the node has no dump record."""
        for node in self._graph.nodes:
            if node.node_type in node_types:
                if mapping.get(node.internal_id, None) is not None:
                    continue
                input_node_count = len(node.from_nodes)
                if input_node_count != 1:
                    print(f'{node.name} has {input_node_count} input node(s), cannot copy dump records.')
                    continue

                input_node = next(iter(node.from_nodes))
                recs = mapping.get(input_node.internal_id, None)
                if recs is None:
                    print(f"{node.name}'s input:{input_node.name} has no dump, cannot copy dump records.")
                    continue
                mapping[node.internal_id] = copy.deepcopy(recs)

    @staticmethod
    def _extract_bin_shape_dtype_format(filename):
        """
        Extract shape dtype and format from .bin filename.

        Notes:
            Expecting filename looks like:
                <operator name>_<input/output>_<index>_shape_<shape>_<dtype>_<format>.bin
            e.g:
                conv2-Conv2d--Conv2D_output_0_shape_5_5_6_16_float32_NHWC.bin
        """
        # remove '.bin'
        filename = filename[:-4]

        splited = filename.split('_')
        splited.reverse()

        data_format = splited[0]
        if data_format.lower() == 'defaultformat':
            data_format = None

        dtype = None
        if len(splited) > 1:
            match = DATA_TYPE_PATTERN.search(splited[1].lower())
            if match:
                dtype = DATA_TYPE.get(match.group(1), None)

        try:
            keyword_idx = splited.index('shape')
            if keyword_idx < 3:
                return None, dtype, data_format
        except ValueError:
            return None, dtype, data_format

        shape_str_list = splited[2:keyword_idx]
        shape = []
        for shape_str in reversed(shape_str_list):
            try:
                shape.append(int(shape_str))
            except ValueError:
                return None, dtype, data_format

        return shape, dtype, data_format
