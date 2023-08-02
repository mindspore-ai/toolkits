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

"""
TroubleShooter Module Introduction.

This module provides Python APIs to shoot trouble of MindSpore neural networks.
Users can import the proposal, initialize the ProposalAction object to start analyse,
and use @proposal() to analyse the Traceback error message.
Users can import the snooping, initialize the Snooper object to start debug,
and use @snooping(...) to print the running result information of echo line code of neural networks.
"""
from __future__ import absolute_import

# The import of mindspore and torch has sequential requirements,
# so the framework identification must be carried out first.
from troubleshooter.common import FRAMEWORK_TYPE
from troubleshooter import common, migrator, proposer, tracker, widget
from troubleshooter.migrator import save
from troubleshooter.widget import fix_random
from troubleshooter.proposer import ProposalAction as proposal
from troubleshooter.tracker import Tracker as tracking
from troubleshooter.version import __version__
