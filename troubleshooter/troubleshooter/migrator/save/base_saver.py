import os
from collections import OrderedDict

import numpy as np


class SaveTensorBase:
    def _clear_cnt(self):
        pass

    def _numpy(self, data):
        pass

    def _shape(self, data):
        pass

    def _split_path_and_name(self, file, sep=os.sep):
        if file[-1] == sep:
            raise ValueError(f"For 'ts.save', the type of argument 'file' must be a valid filename, but got {file}")
        path, name = "", ""
        for c in file:
            if c == sep:
                path = path + name + sep
                name = ""
            else:
                name += c
        return path, name

    def _remove_npy_extension(self, file_name):
        has_extension = False
        extension = ""
        file_name_without_extension = ""

        for char in file_name:
            if char == ".":
                file_name_without_extension += extension
                has_extension = True
                extension = "."
            elif has_extension:
                extension += char
            else:
                file_name_without_extension += char

        if extension == ".npy":
            return file_name_without_extension
        else:
            return file_name

    def _iterate_items(self, data):
        if isinstance(data, (dict, OrderedDict)):
            return data.items()
        elif isinstance(data, (list, tuple)):
            return enumerate(data)
        else:
            raise TypeError("Unsupported data type")

    def _save_tensors(self, path, name, data, auto_id=True, suffix=None):
        if isinstance(data, (list, tuple, dict, OrderedDict)):
            for key, val in self._iterate_items(data):
                item_name = name if name else "tensor_" + str(self._shape(val))
                if auto_id:
                    np.save(f"{path}{int(self._cnt)}_{item_name}_{key}_{suffix}" if suffix else
                            f"{path}{int(self._cnt)}_{item_name}_{key}", self._numpy(val))
                else:
                    np.save(f"{path}{item_name}_{key}_{suffix}" if suffix else
                            f"{path}{item_name}_{key}", self._numpy(val))
        else:
            name = name if name else "tensor_" + str(self._shape(data))
            if auto_id:
                np.save(f"{path}{int(self._cnt)}_{name}_{suffix}" if suffix else
                        f"{path}{int(self._cnt)}_{name}", self._numpy(data))
            else:
                np.save(f"{path}{name}_{suffix}" if suffix else f"{path}{name}",
                        self._numpy(data))

    def _handle_path(self, file):
        if file:
            path, name = self._split_path_and_name(file)
            name = self._remove_npy_extension(name)
        else:
            path, name = "", ""
        return path, name
