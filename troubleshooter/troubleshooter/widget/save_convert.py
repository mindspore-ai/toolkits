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
from troubleshooter import FRAMEWORK_TYPE

if "mindspore" in FRAMEWORK_TYPE:
    import re
    import os
    import stat
    import mindspore as ms
    import numpy as np
    from tqdm import tqdm
    from pathlib import Path
    from troubleshooter import log as logger
    from troubleshooter.common.util import isfile_check, validate_and_normalize_path
    from troubleshooter.migrator.save import SAVE_NAME_MARK


    class _ConvertData:
        """
        Offline conversion of files saved by ts.save to npy files.
        Using a finite state machine.
        """
        tensor_to_np_type = {"Int8": np.int8, "UInt8": np.uint8, "Int16": np.int16, "UInt16": np.uint16,
                             "Int32": np.int32, "UInt32": np.uint32, "Int64": np.int64, "UInt64": np.uint64,
                             "Float16": np.float16, "Float32": np.float32, "Float64": np.float64, "Bool": np.bool_, "str": "U"}

        def __init__(self, output_path):
            self.states = ['pepare', 'parse_data']
            self.state = self.states[0]
            self.path = Path()
            self.output_path = Path(output_path)
            if not self.output_path.exists():
                self.output_path.mkdir(mode=stat.S_IRWXU, parents=True, exist_ok=True)
            self.name = ""
            self.auto_id = -1

            self.events = dict()
            self.actions = {state: dict() for state in self.states}
            self._add_event('pepare', self.pepare)
            self._add_action('pepare', 'pepare', self.skip)
            self._add_action('pepare', 'parse_data', self.skip)
            self._add_event('parse_data', self.parse_data)
            self._add_action('parse_data', 'pepare', self.reset)

        def _add_event(self, state, handler):
            self.events[state] = handler

        def _add_action(self, cur_state, next_state, handler):
            self.actions[cur_state][next_state] = handler

        def run(self, inputs):
            try:
                new_state, outputs = self.events[self.state](inputs)
                self.actions[self.state][new_state](outputs)
                self.state = new_state
            except Exception as e:
                logger.user_error("Because the data was incomplete when saving, "
                                  "the parsing of this data failed. "
                                  f"The current status is {self.state}, and the exception is {e}.")
                self.reset()
                new_state, outputs = 'pepare', None
            return new_state

        def pepare(self, inputs):
            if isinstance(inputs, str) and inputs.startswith(SAVE_NAME_MARK):
                self.parse_name(inputs)
                converted_data = self._convert_str2ndarray(inputs)
                if converted_data is None:
                    return self.states[1], inputs
                else:
                    self._save_data(converted_data)
                    return self.states[0], None
            return self.states[0], None

        def parse_name(self, data_info):
            data_info = data_info.split('\n')[0]
            self.name = data_info[len(SAVE_NAME_MARK):]
            self.auto_id += 1

        def parse_data(self, data):
            np_data = self._convert2ndarray(data)
            self._save_data(np_data)
            return self.states[0], None

        def _parse_shape(self, data):
            if data:
                return tuple(map(int, data.split()))
            else:
                return tuple()

        def _convert_str2ndarray(self, input_string):
            """
            Convert GE string Tensor to ndarray
            """
            pattern = r'Tensor\(shape=(.*?), dtype=(.*?), value=(.*?)\)'

            match = re.search(pattern, input_string, re.DOTALL)
            if match:
                shape_str = match.group(1).strip()
                dtype_str = match.group(2).strip()
                value_str = match.group(3).strip().replace(
                    '[', '').replace(']', '')
                shape = self._parse_shape(shape_str[1:-1])
                dtype = self.tensor_to_np_type[dtype_str]
                value = np.fromstring(
                    value_str, dtype=dtype, sep=' ').reshape(shape)
                return value
            return None

        def _convert2ndarray(self, data):
            if isinstance(data, ms.Tensor):
                return data.asnumpy()
            elif isinstance(data, str):
                return self._convert_str2ndarray(data)
            else:
                raise ValueError(f"Data type error, got type is {type(data)}")

        def _save_data(self, data):
            self.name = str(self.auto_id) + "_" + self.name
            file_path = str(self.output_path/self.name)
            if not file_path.endswith(".npy"):
                file_path = file_path + ".npy"
            if os.path.exists(file_path):
                os.chmod(file_path, stat.S_IWUSR)
            np.save(file_path, data)
            os.chmod(file_path, stat.S_IRUSR)

        def reset(self, *args, **kwargs):
            self.name = ""

        def skip(*args, **kwargs):
            return None

    def save_convert(file, output_path):
        """
        convert save use mindspore print_file to npy files
        """
        isfile_check(file, 'file')
        validate_and_normalize_path(output_path)
        data = ms.parse_print(file)
        output_path = Path(output_path)
        converter = _ConvertData(output_path)
        logger.info("Print dump data total len is ", len(data))
        for item in tqdm(data):
            converter.run(item)
        logger.user_attention(
            f"Convert data has been saved in {output_path.absolute()}")
else:
    def save_convert(file, output_path='npy_files'):
        """
        convert save use mindspore print_file to npy files.
        If there is no mindspore in the environment, it is not supported.
        """
        raise ValueError("There is no 'mindspore' package in the current environment, "
                         "and 'convert_data' does not support calling.")
