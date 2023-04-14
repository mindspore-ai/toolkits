# mindspore callback function
import numpy as np
import mindspore
from mindspore import nn, context, Tensor
from mindspore.common import initializer
from tests.util import delete_file, file_and_key_match
import troubleshooter as ts

"""
def test_front_callback():
    class Print_info(mindspore.Callback):
        def step_end(self, run_context):
            cb_params = run_context.original_args()
            print("step_num: ", cb_params.cur_step_num)
            raise ValueError("call back throw value error")

    delete_file("/tmp/")
    with ts.proposal(write_file_path="/tmp/"):
        print_cb = Print_info()
        data = {"x": np.float32(np.random.rand(64, 10)), "y": np.random.randint(0, 5, (64,))}
        dataset = ds.NumpySlicesDataset(data=data).batch(32)
        net = nn.Dense(10, 5)
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        optim = nn.Momentum(net.trainable_params(), 0.01, 0.9)
        model = mindspore.Model(net, loss_fn=loss, optimizer=optim)
        model.train(1, dataset, callbacks=print_cb)
    assert file_and_key_match("/tmp/", "dataset_id_7")
"""


def test_front_initializer():
    delete_file("/tmp/")
    data = Tensor(np.zeros([1, 2, 3]), mindspore.float32)
    with ts.proposal(write_file_path="/tmp/"):
        tensor1 = initializer(data, [1, 2, 3], mindspore.float32)
    assert file_and_key_match("/tmp/", "front_id_5")


def test_front_nn_gru():
    context.set_context(mode=context.GRAPH_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.gru = nn.GRU(10, 16, 1000, has_bias=True, batch_first=True, bidirectional=False)

        def construct(self, x, h0):
            output = self.gru(x, h0)
            return output

    delete_file("/tmp/")
    with ts.proposal(write_file_path="/tmp/"):
        net = Net()
        x = Tensor(np.ones([3, 5, 10]).astype(np.float32))
        h0 = Tensor(np.ones([1 * 1000, 3, 16]).astype(np.float32))
        output, hn = net(x, h0)
        print('output', output.shape)
    assert file_and_key_match("/tmp/", "compiler_id_13")


def test_nn_softmax_crossentropywithlogits():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False)

        def construct(self, logits, labels):
            output = self.loss(logits, labels)
            return output

    @ts.proposal(write_file_path="/tmp/")
    def main():
        net = Net()
        logits = Tensor(np.array([[[2, 4, 1, 4, 5], [2, 1, 2, 4, 3]]]), mindspore.float32)
        labels = Tensor(np.array([[[0, 0, 0, 0, 1], [0, 0, 0, 1, 0]]]).astype(np.float32))
        print(logits.shape, labels.shape)
        out = net(logits, labels)
        print('out', out)

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "operator_id_14")


def test_front_context_case():
    delete_file("/tmp/")
    with ts.proposal(write_file_path="/tmp/"):
        # import pdb
        # pdb.set_trace()
        context.set_context(device_target='Ascend')
    assert file_and_key_match("/tmp/", "front_id_4")