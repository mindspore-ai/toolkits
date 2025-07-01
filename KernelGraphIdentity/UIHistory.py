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
