import numpy as np
import mindspore
from mindspore import context, nn, Tensor, ops, ms_function
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.parameter import ParameterTuple, Parameter
from mindspore.common.initializer import initializer, XavierUniform
from tests.util import delete_file, file_and_key_match
import troubleshooter as ts

context.set_context(mode=mindspore.GRAPH_MODE, device_target="CPU")


def test_compiler_cls_customization():
    class LayerParams:
        def __init__(self, dtype: str):
            self._type = dtype

        def get_weights(self, shape):
            nn_param = initializer(XavierUniform(), shape, mindspore.float32)
            nn_param = mindspore.Parameter(nn_param)
            return nn_param

    class MyCell(nn.Cell):
        def __init__(self):
            super().__init__()
            self._fc_params = LayerParams("fc")
            self.matmul = ops.MatMul()

        def _fc(self, inputs, output_size):
            width = inputs.shape[-1]
            weight = self._fc_params.get_weights((width, output_size))
            return weight

        def construct(self, x, output_size):
            weight = self._fc(x, output_size)
            output = self.matmul(x, weight)
            return output

    @ts.proposal(write_file_path="/tmp/")
    def test_cls_customization():
        net = MyCell()
        inputs = Tensor(np.ones((2, 4), dtype=np.float32))
        outputs = net(inputs, 5)
        print(outputs.shape)

    delete_file("/tmp/")
    test_cls_customization()
    assert file_and_key_match("/tmp/", "compiler_id_16")


def test_compiler_kmetatypenone():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            return x + self.y

    @ts.proposal(write_file_path="/tmp/")
    def main():
        net = Net()
        out = net(1)

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "compiler_id_12")


def test_compiler_no_self():
    class Net(nn.Cell):
        def construct(x):
            return x

    @ts.proposal(write_file_path="/tmp/")
    def main():
        net = Net()
        out = net(2)
        print("out=", out)

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "compiler_id_1")


def test_compiler_no_self_2():
    class Net_LessInput(nn.Cell):
        def construct(self, x, y):
            return x + y

    @ts.proposal(write_file_path="/tmp/")
    def less_input_case():
        net = Net_LessInput()
        out = net(1)
        print(out)

    delete_file("/tmp/")
    less_input_case()
    assert file_and_key_match("/tmp/", "compiler_id_2")


def test_compiler_no_self_3():
    class Net_MoreInput(nn.Cell):
        def construct(self, x):
            return x

    @ts.proposal(write_file_path="/tmp/")
    def more_input_case():
        net = Net_MoreInput()
        out = net(1, 2)
        print(out)

    delete_file("/tmp/")
    more_input_case()
    assert file_and_key_match("/tmp/", "compiler_id_4")
