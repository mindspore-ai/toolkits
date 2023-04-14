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

"""All proposers."""
from troubleshooter.proposer.allproposers.default_proposer import DefaultProposer
from troubleshooter.proposer.allproposers.front_proposer import FrontProposer
from troubleshooter.proposer.allproposers.compiler_proposer import CompilerProposer
from troubleshooter.proposer.allproposers.dataset_proposer import DatasetProposer
from troubleshooter.proposer.allproposers.operators_proposer import OperatorsProposer
from troubleshooter.proposer.allproposers.compiler_scene_proposer import CompilerSceneProposer
from troubleshooter.proposer.allproposers.dataset_scene_proposer import DatasetSceneProposer
from troubleshooter.proposer.allproposers.operators_scene_proposer import OperatorsSceneProposer
from troubleshooter.proposer.allproposers.vm_proposer import VmProposer

__all__ = ["DefaultProposer", "FrontProposer", "CompilerProposer", "DatasetProposer",
           "OperatorsProposer", "VmProposer", "CompilerSceneProposer", "OperatorsSceneProposer",
           "DatasetSceneProposer", "proposer_list"]
proposer_list = ['front', 'dataset', 'compiler', 'operators', 'vm', 'compiler_scene', 'operators_scene',
                 'dataset_scene']
