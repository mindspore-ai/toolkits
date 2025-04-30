class MemoryAttribute:
    def __init__(self):
        self._dtype = ""
        self._shape: list[int] = []
        self._exist_stride_offset = False  # stride和offset是否存在
        self._stride: list[int] = []
        self._offset = 0

    def set_dtype(self, dtype: str):
        self._dtype = dtype

    def set_shape(self, shape: list[int]):
        self._shape = shape

    def set_exist_stride_offset(self):
        self._exist_stride_offset = True

    def set_stride(self, stride: list[int]):
        self._stride = stride

    def set_offset(self, offset: int):
        self._offset = offset

    # BFloat16:[4096, 1, 32, 128]{strdes=[12288, 12288, 384, 1],offset=0}
    def __repr__(self):
        shape_str = ", ".join(map(str, self._shape))
        res_str = f"{self._dtype}:[{shape_str}]"

        if self._exist_stride_offset:
            stride_str = ", ".join(map(str, self._stride))
            res_str += f"{{strdes=[{stride_str}],offset={self._offset}}}"

        return res_str

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        if isinstance(other, MemoryAttribute):
            return self.__repr__() == other.__repr__()
        return False
