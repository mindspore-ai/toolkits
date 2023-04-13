import os
import numpy as np
import platform
from mindspore import context, Tensor
from mindspore.common import dtype as mstype
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore.ops import DataType, CustomRegOp


class AOTSingleOutputNet(Cell):
    def __init__(self, func, out_shapes, out_types, reg=None):
        super(AOTSingleOutputNet, self).__init__()

        self.program = ops.Custom(func, out_shapes, out_types, "aot", reg_info=reg)

    def construct(self, x, y):
        return self.program(x, y)


def get_file_path_cpu(cc, so):
    dir_path = "../data"
    cmd = "g++ -std=c++17 --shared -fPIC -o " + dir_path + "/aot_test_files/" + so + " " + dir_path + \
          "/aot_test_files/" + cc
    func_path = dir_path + "/aot_test_files/" + so
    return cmd, func_path


def aot_single_output(get_file_path, source, execf, reg):
    shape = (4, 5)
    input_x = np.random.normal(0, 1, shape).astype(np.float32)
    input_y = np.random.normal(0, 1, shape).astype(np.float32)
    cmd, func_path = get_file_path(source, execf)
    print("func_path: ", func_path)
    try:
        test = AOTSingleOutputNet(func_path + ":CustomAdd", (shape,), (mstype.float32,), reg)
        output = test(Tensor(input_x), Tensor(input_y))[0]
    except Exception as e:
        if os.path.exists(func_path):
            os.remove(func_path)
        raise e
    os.remove(func_path)
    print("output:", output)
    assert np.allclose(input_x + input_y, output.asnumpy(), 0.001, 0.001)


add_cpu_info = CustomRegOp() \
    .input(0, "x1") \
    .input(1, "x2") \
    .output(0, "y") \
    .dtype_format(DataType.None_None, DataType.None_None, DataType.None_None) \
    .target("CPU") \
    .get_op_info()


def test_aot_single_output_cpu():
    """
    Feature: custom aot operator, multiple inputs, single output, CPU, GRAPH_MODE
    Description: pre-compile xxx.cc to xxx.so, custom operator launches xxx.so
    Expectation: nn result matches numpy result
    """
    sys = platform.system()
    if sys == "Windows":
        return

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    aot_single_output(get_file_path_cpu, "add.cc", "add.so", add_cpu_info)


if __name__ == "__main__":
    test_aot_single_output_cpu()
