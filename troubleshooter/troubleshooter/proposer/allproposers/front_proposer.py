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
"""Me Front Proposer"""
import traceback
from troubleshooter.proposer.allproposers.base_proposer import Proposer
from troubleshooter.proposer.knowledge_base.front_exp_lib import front_experience_list_cn, \
    general_front_experience_list_cn


class FrontProposer(Proposer):
    """The Front proposer."""

    def __init__(self):
        super().__init__()
        self.experience_list = front_experience_list_cn
        self.scene_experience_list = general_front_experience_list_cn

    def analyze(self, exc_type, exc_value, traceback_obj):
        """
        Get the proposal from proposer.

        Args:
            options: options for proposer analysis

        Returns:
            dict, the proposal from proposer instance.
        """
        error_message = traceback.format_exc()
        result = super().base_case_analyze(error_message, self.experience_list)
        if not result:
            result = super().base_scene_analyze(error_message, self.scene_experience_list)
        return result
