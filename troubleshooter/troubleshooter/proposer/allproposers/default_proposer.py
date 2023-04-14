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

"""The default proposer."""
from troubleshooter.proposer.allproposers.base_proposer import Proposer
from troubleshooter.proposer.knowledge_base.common_exp_lib import default_experience_list_cn


class DefaultProposer(Proposer):
    """The default proposer."""

    def analyze(self, exc_type, exc_value, traceback_obj):
        """
        Get the proposal from proposer.

        Args:
            options: options for proposer analysis

        Returns:
            dict, the proposal from proposer instance.
        """

        return default_experience_list_cn[-1]
