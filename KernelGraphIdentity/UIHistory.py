

import os
import shutil
import dill
import sys
import copy
from ExecutingOrderPreCheckUI import ExecutingOrderPreCheckUI


class UIHistory:
    def __init__(self, max_history: int = 10):
        self._max_history: int = max_history  # 最大历史记录数量
        self._history_states: list[ExecutingOrderPreCheckUI] = []

    def save_ui_state(self, ui: ExecutingOrderPreCheckUI):
        state = copy.deepcopy(ui)
        if len(self._history_states) >= self._max_history:
            self._history_states.pop(0)  # 移除最老的
        self._history_states.append(state)

    def load_last_ui_state(self) -> ExecutingOrderPreCheckUI | None:
        if not self._history_states:
            return None
        return self._history_states.pop()

    def _clear_history_for_dill(self):
        if os.path.exists(self._history_dir):
            shutil.rmtree(self._history_dir)
    def _init_for_dill(self):
        """
        写入文件性能太差
        """
        self._bash_path = os.path.abspath(".")
        if getattr(sys, 'frozen', False):
            self._bash_path = sys._MEIPASS  # PyInstaller临时解压目录
        self._history_dir = os.path.join(self._bash_path, "ui_history")

        self._clear_history_for_dill()  # 初始时清除上一次程序运行的历史
        os.makedirs(self._history_dir, exist_ok=True)

        self._history_file_num = 0

    def _get_file_path(self, file_index: int) -> str:
        file_path = os.path.join(self._history_dir, f"state_{file_index}.pkl")
        return file_path

    def save_ui_state_for_dill(self, ui: ExecutingOrderPreCheckUI):
        file_path = self._get_file_path(self._history_file_num)
        with open(file_path, 'wb') as f:
            dill.dump(ui, f)
        self._history_file_num += 1

    def load_last_ui_state_for_dill(self) -> ExecutingOrderPreCheckUI | None:
        if self._history_file_num == 0:  # 没有历史记录
            return None
        file_index = self._history_file_num - 1
        file_path = self._get_file_path(file_index)
        if not os.path.exists(file_path):  # 历史记录文件不存在
            return None
        with open(file_path, 'rb') as f:
            saved_state = dill.load(f)
        os.remove(file_path)
        self._history_file_num -= 1
        return saved_state
