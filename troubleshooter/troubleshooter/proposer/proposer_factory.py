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

"""The shooter factory."""
import threading
import troubleshooter.proposer.allproposers as proposer_module


class ProposerFactory:
    """The proposer factory is used to create proposer special instance."""
    _lock = threading.Lock()
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    @classmethod
    def instance(cls):
        """The factory instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_proposer(self, proposer_type, *args):
        """
        Get the specified proposer according to the proposer type.

        Args:
            proposer_type (str): The proposer type.
            args (list): The parameters required for the specific proposer class.

        Returns:
            proposer, the specified proposer instance.

        Examples:
            >>> proposer_type = 'default'
            >>> proposer = ProposerFactory.instance().get_proposer(proposer_type)
        """

        proposer_instance = None
        sub_names = proposer_type.split('_')
        proposer_class_name = ''.join([name.capitalize() for name in sub_names])
        proposer_class_name += 'Proposer'
        if hasattr(proposer_module, proposer_class_name):
            proposer_instance = getattr(proposer_module, proposer_class_name)(*args)
        return proposer_instance
