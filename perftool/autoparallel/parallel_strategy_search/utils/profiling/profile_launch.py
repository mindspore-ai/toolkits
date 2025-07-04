from utils.common import LLAMA_PATTERN_REG_SHELL
from utils.logger import logger
import subprocess
import os
import re
import time
class ProfileLaunch:
    # 自动执行profile
    def __init__(self, profile_configs, para):
        self.profile_configs = profile_configs
        self.para = para
        pass

    def profile_launch(self, profile_file_dir):
        # 遍历指定目录下的所有文件
        for root, _, files in os.walk(profile_file_dir):
            for file in files:
                pattern = LLAMA_PATTERN_REG_SHELL
                match = re.match(pattern, file)
                if not match:
                    continue
                profile_file_path = os.path.join(root, file)
                self.run(profile_file_path)

    def run(self, profile_file_dir):
        cmd = ["bash", profile_file_dir]
        logger.info(f"profile command: {cmd}")

        process = subprocess.run(
            cmd,
            preexec_fn=os.setpgrp,
            check=False,
        )
        # 为避免profile子进程未结束生成profile文件失败，增加sleep
        time.sleep(60)
        return_code = process.returncode
        logger.info("Last job returns %d.", return_code)