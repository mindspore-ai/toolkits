from collections import defaultdict
import mindtorch.torch.nn as nn

module_count = defaultdict(int)
g_stop_hook = False


class HOOKModule(nn.Module):

    def __init__(self, hook) -> None:
        super(HOOKModule, self).__init__()
        self.changed_status = False

        global g_stop_hook
        if not g_stop_hook:
            g_stop_hook = True
            prefix = ""
            self.changed_status = True
            if hasattr(self, "prefix_op_name_"):
                prefix = self.prefix_op_name_

            module_count[prefix] += 1
            prefix = prefix + str(module_count[prefix] - 1) + '_'

            self.register_forward_hook(hook(prefix + "forward"))

    def __call__(self, *args, **kwargs):
        try:
            out = super(HOOKModule, self).__call__(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            if self.changed_status:
                self.changed_status = False
                global g_stop_hook
                g_stop_hook = False
        return out