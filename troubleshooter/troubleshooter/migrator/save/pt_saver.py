import numpy as np
import os
import torch

_g_save_cnt = 0


def save(file, data):
    def numpy(data):
        if torch.is_tensor(data):
            return data.cpu().detach().numpy()
        else:
            raise TypeError(f"For 'ts.save', the type of argument 'data' must be mindspore.Tensor or torch.tensor, " \
                            f"but got {type(data)}")

    path, name = os.path.split(file)
    if not name:
        raise ValueError(f"For 'ts.save', the type of argument 'file' must be a valid filename, but got {file}")
    global _g_save_cnt
    np.save(f"{path}{int(_g_save_cnt)}_{name}", numpy(data))
