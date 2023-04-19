import numpy as np
import mindspore
from mindspore import context, nn, Tensor, ops, ms_function
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.parameter import ParameterTuple, Parameter
from mindspore.common.initializer import initializer
from tests.util import delete_file, file_and_key_match
import troubleshooter as ts

context.set_context(mode=mindspore.GRAPH_MODE, device_target="CPU")


def test_compile_paramtertuple():
    class FullyConnectedNet(nn.Cell):
        def __init__(self, input_size, hidden_size, output_size):
            super(FullyConnectedNet, self).__init__(auto_prefix=False)
            self.linear1 = nn.Dense(input_size, hidden_size, weight_init="XavierUniform")
            self.linear2 = nn.Dense(hidden_size, output_size, weight_init="XavierUniform")
            self.relu = nn.ReLU()

        def construct(self, x):
            x = self.relu(self.linear1(x))
            x = self.linear2(x)
            return x

    class EmaUpdate(nn.Cell):
        def __init__(self, policy_net, target_net):
            super(EmaUpdate, self).__init__()
            self.policy_param = ParameterTuple(policy_net.get_parameters())
            self.target_param = ParameterTuple(target_net.get_parameters())
            self.step = Parameter(initializer(0, [1]), name='step', requires_grad=False)
            self.assignadd = P.AssignAdd()

        def construct(self):
            return self.assignadd(self.step, 1)

    @ts.proposal(write_file_path="/tmp/")
    def test_target_update():
        policy_net = FullyConnectedNet(4, 100, 2)
        target_net = FullyConnectedNet(4, 100, 2)

        ema_update = EmaUpdate(policy_net, target_net)
        # print(ema_update)

    delete_file("/tmp/")
    test_target_update()
    assert file_and_key_match("/tmp/", "compiler_id_18")


def test_compiler_commit_error():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.n = 5.0

        def construct(self, x):
#       x + 2.0
            return x + self.n  # commit after code

    @ts.proposal(write_file_path="/tmp/")
    def main():
        net = Net()
        out = net(1)
        print(out)

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "compiler_id_17")


def test_compiler_exceed_call_depth():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.concat = ops.Concat()

        def construct(self, x):
            n = 1000
            input_x = ()
            for i in range(n):
                input_x += (x,)
            output = self.concat(input_x)
            return output

    @ts.proposal(write_file_path="/tmp/")
    def demo():
        net = Net()
        x = Tensor(np.random.rand(1, 4, 16, 16), mindspore.float32)
        output = net(x)

    delete_file("/tmp/")
    demo()
    assert file_and_key_match("/tmp/", "compiler_id_13")


def test_compiler_func_params():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = ops.Add()
            self.sub = ops.Sub()
            self.mul = ops.Mul()
            self.div = ops.Div()

        def func(x, y):
            return self.div(x, y)

        def construct(self, x, y):
            a = self.sub(x, 1)
            b = self.add(a, y)
            c = self.mul(b, self.func(a, a, b))
            return c

    @ts.proposal(write_file_path="/tmp/")
    def main():
        input1 = Tensor(3, mstype.float32)
        input2 = Tensor(2, mstype.float32)
        net = Net()
        out = net(input1, input2)
        print(out)

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "compiler_id_6")


def test_compiler_grad_join_fail():
    def func(a, b):
        return a, b

    grad = ops.GradOperation(get_by_list=False, sens_param=True)

    @ms_function()
    def test_grad(x, y, sens):
        sens_i = ops.Fill()(ops.DType()(x), ops.Shape()(x), sens)
        a = grad(func)(x, y, sens_i)
        return a

    @ts.proposal(write_file_path="/tmp/")
    def grad_join_fail():
        x = Tensor([1.0])
        y = Tensor([2.0])
        test_grad(x, y, 1.0)

    delete_file("/tmp/")
    grad_join_fail()
    assert file_and_key_match("/tmp/", "compiler_id_10")


def test_compiler_jit_fallback():
    @ms_function
    def test_np_add():
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        # z = Tensor(np.add(x, y))
        return np.add(x, y)

    @ts.proposal(write_file_path="/tmp/")
    def main():
        np_add_res = test_np_add()
        print(np_add_res)

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "compiler_id_14")


def test_compiler_shape_join_failed():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = ops.ReLU()
            self.reducesum = ops.ReduceSum()

        def construct(self, x, a, b):
            if a > b:
                return self.relu(x)  # shape: (2, 3, 4, 5), dtype:Float32
            else:
                return self.reducesum(x)  # shape:(), dype: Float32

    @ts.proposal(write_file_path="/tmp/")
    def main():
        input_x = Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
        input_a = Tensor(2, mindspore.float32)
        input_b = Tensor(6, mindspore.float32)
        net = Net()
        out = net(input_x, input_a, input_b)

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "compiler_id_9")


def test_compiler_type_join_failed():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = ops.ReLU()
            self.cast = ops.Cast()

        def construct(self, x, a, b):
            if a > b:
                return self.relu(x)  # shape: (2, 3, 4, 5), dtype:Float32
            else:
                return self.cast(self.relu(x), mindspore.float16)  # shape: (2, 3, 4, 5)， dtype:Float16

    @ts.proposal(write_file_path="/tmp/")
    def main():
        input_x = Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
        input_a = Tensor(2, mindspore.float32)
        input_b = Tensor(6, mindspore.float32)
        net = Net()
        out_me = net(input_x, input_a, input_b)

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "compiler_id_8")


def test_compiler_with_as_type_join_failed():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = ops.ReLU()
            self.cast = ops.Cast()

        def construct(self, x, a, b):
            if a > b:
                return self.relu(x)  # shape: (2, 3, 4, 5), dtype:Float32
            else:
                return self.cast(self.relu(x), mindspore.float16)  # shape: (2, 3, 4, 5)， dtype:Float16

    def main():
        input_x = Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
        input_a = Tensor(2, mindspore.float32)
        input_b = Tensor(6, mindspore.float32)
        net = Net()
        with ts.proposal(write_file_path="/tmp/") as proposal:
            out_me = net(input_x, input_a, input_b)

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "compiler_id_8")


def test_compiler_get_context():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = ops.Add()
            self.sub = ops.Sub()
            self.mul = ops.Mul()
            self.div = ops.Div()

        def func(x, y):
            return self.div(x, y)

        def construct(self, x, y):
            context.get_context("device_target")
            a = self.sub(x, 1)
            b = self.add(a, y)
            c = self.mul(b, self.func(a, b))
            return c

    @ts.proposal(write_file_path="/tmp/")
    def test1():
        input1 = Tensor(3, mstype.float32)
        input2 = Tensor(2, mstype.float32)
        net = Net()
        out = net(input1, input2)
        print(out)

    delete_file("/tmp/")
    test1()
    assert file_and_key_match("/tmp/", "compiler_id_19")


def test_compiler_get_context_2():
    class Net1(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = ops.Add()
            self.sub = ops.Sub()
            self.mul = ops.Mul()
            self.div = ops.Div()

        def func(x, y):
            context.get_context("device_target")
            return self.div(x, y)

        def construct(self, x, y):
            a = self.sub(x, 1)
            b = self.add(a, y)
            c = self.mul(b, self.func(a, b))
            return c

    @ts.proposal(write_file_path="/tmp/")
    def test2():
        input1 = Tensor(3, mstype.float32)
        input2 = Tensor(2, mstype.float32)
        net = Net1()
        out = net(input1, input2)
        print(out)

    delete_file("/tmp/")
    test2()
    assert file_and_key_match("/tmp/", "compiler_id_19")


def test_compiler_get_context_3():
    class Net2(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = ops.Add()
            self.sub = ops.Sub()
            self.mul = ops.Mul()
            self.div = ops.Div()

        def func(x, y):
            return self.div(x, y)

        def construct(self, x, y):
            try:
                context.set_context(mode=mindspore.GRAPH_MODE)
                a = self.sub(x, 1)
                b = self.add(a, y)
                c = self.mul(b, self.func(a, b))
            except Exception as e:
                print(e)
            return c

    @ts.proposal(write_file_path="/tmp/")
    def test3():
        input1 = Tensor(3, mstype.float32)
        input2 = Tensor(2, mstype.float32)
        net = Net2()
        out = net(input1, input2)
        print(out)

    delete_file("/tmp/")
    test3()
    assert file_and_key_match("/tmp/", "compiler_id_19")


def test_compiler_less_input():
    class Net(nn.Cell):
        def construct(x):
            return x

    @ts.proposal(write_file_path="/tmp/")
    def main():
        net = Net()
        out = net(2)

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "compiler_id_1")


def test_compiler_less_input_1():
    class Net_LessInput(nn.Cell):
        def construct(self, x, y):
            return x + y

    @ts.proposal(write_file_path="/tmp/")
    def less_input_case():
        context.set_context(mode=mindspore.PYNATIVE_MODE, pynative_synchronize=True)
        net = Net_LessInput()
        out = net(1)
        print(out)

    delete_file("/tmp/")
    less_input_case()
    assert file_and_key_match("/tmp/", "compiler_id_3")


def test_compiler_less_input_2():
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
    assert file_and_key_match("/tmp/", "compiler_id_5")


def test_compiler_more_input():
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
    assert file_and_key_match("/tmp/", "common_id_1")


def test_compiler_more_input_2():
    class Net_LessInput(nn.Cell):
        def construct(self, x, y):
            return x + y

    @ts.proposal(write_file_path="/tmp/")
    def less_input_case():
        context.set_context(mode=mindspore.PYNATIVE_MODE, pynative_synchronize=True)
        net = Net_LessInput()
        out = net(1)
        print(out)

    delete_file("/tmp/")
    less_input_case()
    assert file_and_key_match("/tmp/", "compiler_id_3")


def test_compiler_more_input_3():
    class Net_MoreInput(nn.Cell):
        def construct(self, x):
            return x

    @ts.proposal(write_file_path="/tmp/")
    def more_input_case():
        context.set_context(mode=mindspore.PYNATIVE_MODE, pynative_synchronize=True)
        net = Net_MoreInput()
        out = net(1, 2)
        print(out)

    delete_file("/tmp/")
    more_input_case()
    assert file_and_key_match("/tmp/", "compiler_id_5")


def test_compiler_staticmap_1():
    # 'jit_class' need that mindspore version 2.0.0-alpha or later
    @mindspore.jit_class
    class InnerNet:
        def __init__(self, val):
            self.number = val + 3

    class Net(nn.Cell):
        def __init__(self, val):
            super(Net, self).__init__()
            self.inner = InnerNet(val)

        def construct(self, x):
            net = InnerNet(x)
            return net.number

    @ts.proposal(write_file_path="/tmp/")
    def staticmap_case():
        mindspore.set_context(mode=mindspore.GRAPH_MODE)
        x = mindspore.Tensor(2, dtype=mindspore.int32)
        net = Net(x)
        out = net(x)
        print(out)

    delete_file("/tmp/")
    staticmap_case()
    assert file_and_key_match("/tmp/", "compiler_id_20")


def test_compiler_staticmap_2():
    class Net(nn.Cell):
        def __init__(self, mode):
            super(Net, self).__init__()
            self.mode = mode

        def construct(self):
            if self.mode == "print_var":
                var = (1, 2, 3)
            for i in var:
                print(1)
            return True

    @ts.proposal(write_file_path="/tmp/")
    def staticmap_case():
        mindspore.set_context(mode=mindspore.GRAPH_MODE)
        net = Net("not_print_var")
        out = net()
        print(out)

    delete_file("/tmp/")
    staticmap_case()
    assert file_and_key_match("/tmp/", "compiler_id_21")


def test_compiler_staticmap_3():
    x = (Tensor(4), Tensor(5), Tensor(6))
    # 'jit' need that mindspore version 2.0.0-alpha or later

    @mindspore.common.jit
    def index_get(y):
        return x[y]

    @ts.proposal(write_file_path="/tmp/")
    def staticmap_case():
        y = Tensor(2)
        out = index_get(y)
        print(out)

    delete_file("/tmp/")
    staticmap_case()
    assert file_and_key_match("/tmp/", "compiler_id_22")


def test_compiler_staticmap_4():
    x = (Tensor(4), Tensor(5), Tensor(6))

    class IndexGet(nn.Cell):
        def construct(self, x, y):
            return x[y]

    @ts.proposal(write_file_path="/tmp/")
    def staticmap_case():
        context.set_context(mode=context.GRAPH_MODE, save_graphs=True, save_graphs_path="../ir/tupleindextensor/1")
        y = Tensor(2, mindspore.int32)
        print(IndexGet()(x, y))

    delete_file("/tmp/")
    staticmap_case()
    assert file_and_key_match("/tmp/", "compiler_id_23")
