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
"""VM Proposer"""
import traceback
from troubleshooter.common.util import compare_key_words, compare_stack_words
from troubleshooter.common.format_msg import format_error_message
from troubleshooter.proposer.allproposers.base_proposer import Proposer
from troubleshooter.proposer.knowledge_base.vm_exp_lib import vm_experience_list_cn, vm_general_experience_list_cn


class VmProposer(Proposer):
    """The VM proposer."""

    def __init__(self):
        super().__init__()
        self.experience_list = vm_experience_list_cn
        self.general_experience_list = vm_general_experience_list_cn

    def analyze(self, exc_type, exc_value, traceback_obj):
        """
        Vm analyze
        Args:
            exc_type:
            exc_value:
            traceback_obj:

        Returns:

        """
        error_message = str(exc_value) if exc_value else traceback.format_exc()
        msg_dict = format_error_message(error_message)
        result = None

        for experience in self.experience_list:
            key_words = experience.get('Key Log Information')
            python_stack_info = experience.get('Key Python Stack Information')
            cplusplus_stack_info = experience.get('Key C++ Stack Information')
            err_msg = msg_dict.get('error_message')
            cpp_stack = msg_dict.get('C++ Call Stack: (For framework developers)')
            ascend_err_msg = msg_dict.get('Ascend Error Message:')

            if (compare_key_words(err_msg, key_words) or compare_key_words(ascend_err_msg, key_words)) and \
                    compare_stack_words(err_msg, python_stack_info) and \
                    compare_stack_words(cpp_stack, cplusplus_stack_info):
                result = experience
                break

        if result is None:
            result = super().base_scene_analyze(error_message, self.general_experience_list)

        return result
