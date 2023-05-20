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
"""Tracker"""
import os
from pysnooper.tracer import Tracer
from pysnooper.tracer import get_path_and_source_from_frame
from pysnooper.utils import get_repr_function
from troubleshooter import log as logger

_NO_RESULT = -1
_FRAMEWORK_MS = 1


# pylint: disable=W0703
def get_local_reprs(frame, check_keyword):
    """
    get local reprs
    Args:
        check_keyword: check keyword
        frame: python frame
    """
    for key, value in frame.f_locals.items():
        repr_function = get_repr_function(value, custom_repr=())
        try:
            r = repr_function(value)
        except Exception:
            r = 'REPR FAILED'
        r = r.replace('\r', '').replace('\n', '')
        if r.find(check_keyword) != _NO_RESULT and r.find('value=') != _NO_RESULT:
            logger.user_warning("'{}' is detected, Key is {}, Value is {}".format(check_keyword, key, r))
            return True
    return False


class Tracker(Tracer):
    """
    Framework Code Execution Tracking
    Args:
        framework (bool): Framework type
        filter (dict): Tracking filtering
        *args: Tracer args
        **kwargs: Tracer args
    """

    def __init__(self, level=1, **kwargs):
        self.root_file = ""
        self.root_path = ""
        self.level = level
        self.ms_func_blacklist = {"deco": ["ops/primitive.py"], "__init__": ["ops/primitive.py", "common/tensor.py"],
                                  "__new__": ["common/parameter.py"], "__setattr__": ["nn/cell.py"],
                                  "__getattr__": ["nn/cell.py"],
                                  "_get_cache_prim_for_pynative": ["ops/_primitive_cache.py"]}
        self.func_whitelist = kwargs.get('func_wl', None)
        self.path_whitelist = kwargs.get('path_wl', [])
        self.path_blacklist = kwargs.get('path_bl', [])
        self.check_keyword = kwargs.get('check_keyword', None)
        self.check_mode = kwargs.get('check_mode', 1)
        self.framework = kwargs.get('framework', 1)
        self.event_list = kwargs.get('event_list', [])
        self.depth = kwargs.get('depth', 15)
        self.max_variable_length = kwargs.get('max_variable_length', 150)
        if self.event_list and 'call' in self.event_list and 'return' not in self.event_list:
            self.event_list.append('return')
        if self.level == 2:
            blacklist = {"__setattr__": "nn/cell.py", "__getattr__": "nn/cell.py"}
            self.ms_func_blacklist.update(blacklist)

        self.pop_key_list = ['event_list', 'func_wl', 'path_wl', 'depth', 'max_variable_length', 'check_keyword',
                             'path_bl', 'check_mode', 'framework']
        for key in self.pop_key_list:
            value = kwargs.get(key)
            if value is not None:
                kwargs.pop(key)
        super(Tracker, self).__init__(depth=self.depth, max_variable_length=self.max_variable_length, **kwargs)

    def _frame_filter(self, frame, event):
        if self.framework == _FRAMEWORK_MS:
            return self._frame_filter_ms(frame, event)
        return True

    def _frame_filter_ms(self, frame, event):
        """
        Filter the frame information of
        Args:
            frame: The python frame
            event: Reserved Fields
            arg: Reserved Fields

        Returns (bool): frame information
        """
        result = True

        # line_no = frame.f_lineno
        func_name = frame.f_code.co_name
        source_path = get_path_and_source_from_frame(frame)
        source_path = source_path if not self.normalize else os.path.basename(source_path)

        if self.root_file == "":
            self.root_file = source_path[0]
            self.root_func = func_name
            self.root_path = os.path.abspath(os.path.dirname(self.root_file))
            logger.info("the root_file is {}".format(self.root_file))
            logger.info("the root_func is {}".format(self.root_func))
            logger.info("the root_path is {}".format(self.root_path))
            return result

        logger.info("the path_name is {}".format(source_path[0]))
        logger.info("the func_name is {}".format(func_name))

        # ignore the MindSpore backlist function
        ignore_path_list = self.ms_func_blacklist.get(func_name)
        if ignore_path_list:
            for ignore_path in ignore_path_list:
                logger.debug("the source_path[0] is {},ignore_path is {}".format(source_path[0], ignore_path))
                if source_path[0].find(ignore_path) != _NO_RESULT:
                    return False

        if self.path_blacklist:
            for path in self.path_blacklist:
                if source_path[0].find(path) != _NO_RESULT:
                    return False

        if self.path_whitelist:
            for path in self.path_whitelist:
                return source_path[0].find(path) != _NO_RESULT

        if self.level == 1 and (func_name not in ["construct"] or
                                (source_path[0].find(self.root_path) == _NO_RESULT and
                                 source_path[0].find("container.py") == _NO_RESULT)):
            return False

        if self.level == 2 and (func_name not in ["construct", "_convert_python_data"]):
            return False

        if self.level == 3 and source_path[0].find("site-packages/mindspore") == _NO_RESULT and \
                source_path[0].find(self.root_path) == _NO_RESULT:
            return False

        if self.check_keyword and self.check_mode == 1:
            check_result = get_local_reprs(frame, self.check_keyword)
            if check_result:
                exit(0)
        elif self.check_keyword and self.check_mode == 2:
            get_local_reprs(frame, self.check_keyword)

        if self.event_list and event not in self.event_list:
            return False

        return result

    def trace(self, frame, event, arg):
        if not self._frame_filter(frame, event):
            return None
        return super(Tracker, self).trace(frame, event, arg)
