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
"""Dataset scene Proposer"""
from troubleshooter.proposer.allproposers.base_proposer import Proposer
from troubleshooter.proposer.knowledge_base.dataset_exp_lib import data_general_experience_list_cn


class DatasetSceneProposer(Proposer):
    """The Dataset scene proposer."""

    def __init__(self):
        super().__init__()
        self.experience_list = data_general_experience_list_cn

    def analyze(self, exc_type, exc_value, traceback_obj):
        """

        Args:
            exc_type:
            exc_value:
            traceback_obj:

        Returns:

        """
        error_message = str(exc_value) if exc_value else traceback.format_exc()
        result = super().base_scene_analyze(error_message, self.experience_list)
        return result
