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
"""log module"""
import os
import logging
from troubleshooter.common.util import validate_and_normalize_path

GLOBAL_LOGGER = None


def _get_logger():
    """
    Get logger instance.

    Returns:
        Logger, a logger.
    """
    global GLOBAL_LOGGER
    if GLOBAL_LOGGER:
        return GLOBAL_LOGGER

    log_file_path = os.environ.get("troubleshooter_log_path")
    log_level = os.environ.get("troubleshooter_log_level", logging.WARNING)
    if log_file_path:
        path = validate_and_normalize_path(log_file_path)
        log_file = os.path.join(path, "troubleshooter.log")
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            filename=log_file, filemode='a')
    else:
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)
    GLOBAL_LOGGER = logger
    return logger


def info(msg, *args, **kwargs):
    """
    Log a message with severity 'INFO' on the MindSpore logger.

    Examples:
        >>> from mindspore import log as logger
        >>> logger.info("The arg(%s) is: %r", name, arg)
    """
    _get_logger().info(msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    """
    Log a message with severity 'DEBUG' on the MindSpore logger.

    Examples:
        >>> from mindspore import log as logger
        >>> logger.debug("The arg(%s) is: %r", name, arg)
    """
    _get_logger().debug(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """Log a message with severity 'ERROR' on the MindSpore logger."""
    _get_logger().error(msg, *args, **kwargs)

def user_error(msg, *args, **kwargs):
    """Log a message with severity 'WARNING' on the MindSpore logger."""
    msg = "[*User Error*] " + msg
    _get_logger().error(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    """Log a message with severity 'WARNING' on the MindSpore logger."""
    _get_logger().warning(msg, *args, **kwargs)


def user_warning(msg, *args, **kwargs):
    """Log a message with severity 'WARNING' on the MindSpore logger."""
    msg = "[*User Warning*] " + msg
    _get_logger().warning(msg, *args, **kwargs)


def user_attention(msg, *args, **kwargs):
    """Log a message with severity 'WARNING' on the MindSpore logger."""
    msg = "[*User Attention*] " + msg
    _get_logger().warning(msg, *args, **kwargs)

def critical(msg, *args, **kwargs):
    """Log a message with severity 'CRITICAL' on the MindSpore logger."""
    _get_logger().critical(msg, *args, **kwargs)
