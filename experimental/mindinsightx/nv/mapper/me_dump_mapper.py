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
"""ME Dump file mapper."""
import os
import os.path
import re

import numpy as np

from mindinsightx.nv.mapper.dump_mapper import DumpMapper
from mindinsightx.nv.mapper.dump_mapper import DumpRecord
from mindinsightx.nv.mapper.mappings import mappings


class MeDumpRecord(DumpRecord):
    """ME dump file record."""

    def load(self):
        """Load the dump file content."""
        if self.real_path.endswith('.npy') or self.real_path.endswith('.npz'):
            return np.load(self.real_path)
        if self.dtype is None:
            print(f"dtype of {self.real_path} is unknown, load as float")
            return np.fromfile(self.real_path)
        return np.fromfile(self.real_path, dtype=self.dtype)


class MeDumpMapper(DumpMapper):
    """ME node to dump mapper."""
    def process(self, map_input=True, map_output=True):
        """Build up the mapping."""
        if map_input:
            self._input_map = dict()
        if map_output:
            self._output_map = dict()

        file_type = self._detect_file_type()
        if file_type == 'bin':
            if map_input:
                self._map_bin_dumps('input', self._input_map)
            if map_output:
                self._map_bin_dumps('output', self._output_map)
        elif file_type == 'np':
            if map_input:
                self._map_np_dumps('input', self._input_map)
            if map_output:
                self._map_np_dumps('output', self._output_map)
        else:
            raise RuntimeError(f'No valid dump file type in {self._dump_dir}! .bin, .npy and .npz are supported.')

        if map_input and not self._input_map:
            raise RuntimeError(f'No valid input dump file in {self._dump_dir}!')

        if map_output and not self._output_map:
            raise RuntimeError(f'No valid output dump file in {self._dump_dir}!')

    def _detect_file_type(self):
        """Detect the dump file type."""
        bin_pattern = re.compile(r'.+_(input|output)_.*\.bin$')
        np_pattern = re.compile(r'.+\.(input|output)\..*\.(npy|npz)$')
        dump_dir = os.path.realpath(self._dump_dir)
        for filename in os.listdir(dump_dir):
            if bin_pattern.match(filename):
                real_path = os.path.realpath(os.path.join(self._dump_dir, filename))
                if os.path.isfile(real_path):
                    return 'bin'
            elif np_pattern.match(filename):
                real_path = os.path.realpath(os.path.join(self._dump_dir, filename))
                if os.path.isfile(real_path):
                    return 'np'
        return None

    def _map_bin_dumps(self, io_keyword, mapping):
        """Build up the .bin dump mapping."""
        dump_dir = os.path.realpath(self._dump_dir)
        filenames = os.listdir(dump_dir)
        mapped_flags = [0] * len(filenames)
        for node in self._graph.nodes:
            node_name = node.name.replace('/', '--')
            pattern = re.compile(rf'^{node_name}_{io_keyword}_(\d+).*\.bin$')
            for i, filename in enumerate(filenames):
                if mapped_flags[i]:
                    continue
                match = pattern.match(filename)
                if not match:
                    continue
                real_path = os.path.realpath(os.path.join(dump_dir, filename))
                if os.path.isfile(real_path):
                    shape, dtype, data_format = self._extract_bin_shape_dtype_format(filename)

                    record = MeDumpRecord(real_path, filename, int(match.group(1)), shape, dtype, data_format)

                    records = mapping.get(node.internal_id, None)
                    if records is None:
                        records = []
                        mapping[node.internal_id] = records
                    records.append(record)
                    mapped_flags[i] = 1

        for _, records in mapping.items():
            records.sort(key=lambda x: x.index)
            # records with index larger then 0 may not have data format,
            # use data format of output 0 dump
            data_format_baseline = records[0].data_format
            if data_format_baseline is not None:
                for record in records[1:]:
                    if record.data_format is None:
                        record.data_format = data_format_baseline

    def _map_np_dumps(self, io_keyword, mapping):
        """Build up the .npy/.npz dump mapping."""
        dump_dir = os.path.realpath(self._dump_dir)
        filenames = os.listdir(dump_dir)
        mapped_flags = [0] * len(filenames)
        for node in self._graph.nodes:
            last_name = node.name.split('/')[-1]
            pattern = re.compile(rf'^{node.node_type}\..*{last_name}.*\.(\d+)\.{io_keyword}\.(\d+).*\.(npy|npz)$')
            for i, filename in enumerate(filenames):
                if mapped_flags[i]:
                    continue
                name = filename.split('.')[0]
                # use mappings
                if name in mappings:
                    name = mappings[name]
                fn = '.'.join([name, '.'.join(filename.split('.')[1:])])
                match = pattern.match(fn)
                if not match:
                    continue
                index = int(match.group(2))
                timestamp = int(match.group(1))
                real_path = os.path.realpath(os.path.join(dump_dir, filename))
                self._add_np_record(mapping, node, real_path, filename, index, timestamp)
                mapped_flags[i] = 1

        for _, records in mapping.items():
            records.sort(key=lambda x: x.index)

    def _add_np_record(self, mapping, node, real_path, filename, index, timestamp):
        """Add a .npy/.npz dump record."""
        if os.path.isfile(real_path):
            record = MeDumpRecord(real_path, filename, index, timestamp=timestamp)
            records = mapping.get(node.internal_id, None)
            index_already_exists = False
            if records is None:
                records = []
                mapping[node.internal_id] = records
            else:
                # take the earliest record if the output index already exists
                for i, old_record in enumerate(records):
                    if old_record.index == record.index:
                        if record.timestamp < old_record.timestamp:
                            records[i] = record
                        index_already_exists = True
                        break
            if not index_already_exists:
                records.append(record)
