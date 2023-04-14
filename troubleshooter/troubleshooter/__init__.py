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

"""
TroubleShooter Module Introduction.

This module provides Python APIs to shoot trouble of MindSpore neural networks.
Users can import the proposal, initialize the ProposalAction object to start analyse,
and use @proposal() to analyse the Traceback error message .
Users can import the snooping, initialize the Snooper object to start debug,
and use @snooping(...) to print the running result information of echo line code of neural networks.
"""

from .migrator.diff_handler import TensorRecorder as tensor_recorder
from .migrator.diff_handler import DifferenceFinder as diff_finder
from .migrator.diff_handler import WeightMigrator as weight_migrator
from .proposer import ProposalAction as proposal
from .tracker import Tracker as tracking
