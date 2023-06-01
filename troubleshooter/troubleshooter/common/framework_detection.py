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

"""detect framework"""
from troubleshooter import log as logger

__all__ = ["FRAMEWORK_TYPE"]

FRAMEWORK_TYPE = set()


def _detect_framework():
    global FRAMEWORK_TYPE
    if FRAMEWORK_TYPE:
        return
    try:
        import torch

        FRAMEWORK_TYPE.add("torch")
    except ModuleNotFoundError as e:
        e_msg = e.msg
        no_module_msg = "No module named 'torch'"
        if e_msg != no_module_msg:
            raise e
    except OSError as e:
        e_msg = str(e)
        if "libgomp" in e_msg:
            logger.info(e_msg)
            logger.info("In the environment where both torch and mindspore coexist, "
                        "if mindspore is imported before torch, "
                        "causing torch to fail to locate the libgomp.so file, "
                        "you can try setting the LD_PRELOAD environment variable like "
                        f"'export LD_PRELOAD={e_msg.split(':')[0]}' before running the program to resolve this issue."
                        )
        else:
            raise e

    try:
        import mindspore

        FRAMEWORK_TYPE.add("mindspore")
    except ModuleNotFoundError as e:
        e_msg = e.msg
        no_module_msg = "No module named 'mindspore'"
        if e_msg != no_module_msg:
            raise e


class _ImportError:
    def __init__(self, name, depend) -> None:
        raise ImportError(f"For '{name}', {depend} must be installed, "
                          "please try again after installation.")


class NetDifferenceFinder(_ImportError):
    def __init__(self) -> None:
        super(NetDifferenceFinder, self).__init__(
            "ts.NetDifferenceFinder", "'torch' and 'mindspore'")


class WeightMigrator(_ImportError):
    def __init__(self) -> None:
        super(WeightMigrator, self).__init__("ts.WeightMigrator", "'torch' and 'mindspore'")


def save(*args, **kwargs):
    raise ImportError(f"For 'ts.save', 'torch' or 'mindspore' must be installed, "
                      "please try again after installation.")


_detect_framework()
