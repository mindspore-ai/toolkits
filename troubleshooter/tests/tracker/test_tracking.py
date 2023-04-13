#!/usr/bin/env python
# coding: utf-8

import os
import mindspore
import numpy as np
from mindspore import nn, ops, context, Tensor
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
import troubleshooter as ts
# Download data from open datasets
from download import download
from tests.util import delete_file, file_and_key_match
context.set_context(mode=mindspore.PYNATIVE_MODE)

def test_tacking_for_function_code_2_0_level():
    def main():
        url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip"
        path = download(url, "./", kind="zip", replace=True)

        def datapipe(path, batch_size):
            image_transforms = [
                vision.Rescale(1.0 / 255.0, 0),
                vision.Normalize(mean=(0.1307,), std=(0.3081,)),
                vision.HWC2CHW()
            ]
            label_transform = transforms.TypeCast(mindspore.int32)

            dataset = MnistDataset(path)
            dataset = dataset.map(image_transforms, 'image')
            dataset = dataset.map(label_transform, 'label')
            dataset = dataset.batch(batch_size)
            return dataset

        train_dataset = datapipe('MNIST_Data/train', 64)
        test_dataset = datapipe('MNIST_Data/test', 64)

        class Network(nn.Cell):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.dense_relu_sequential = nn.SequentialCell(
                    nn.Dense(28 * 28, 512),
                    nn.ReLU(),
                    nn.Dense(512, 512),
                    nn.ReLU(),
                    nn.Dense(512, 10)
                )

            # output="/tmp/mindspore_tracking_test.log"
            @ts.tracking(level=1, output="/tmp/mindspore_tracking_test.log")
            def construct(self, x):
                x = self.flatten(x)
                logits = self.dense_relu_sequential(x)
                return logits

        model = Network()

        epochs = 10
        batch_size = 32
        learning_rate = 1e-2

        loss_fn = nn.CrossEntropyLoss()

        optimizer = nn.SGD(model.trainable_params(), learning_rate=learning_rate)

        # @ts.tracking(level=2, depth=15, path_wl=['dataset/engine'])

        def train_loop(model, dataset, loss_fn, optimizer):
            # Define forward function
            def forward_fn(data, label):
                logits = model(data)
                loss = loss_fn(logits, label)
                return loss, logits

            # Get gradient function
            grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

            # Define function of one-step training
            def train_step(data, label):
                (loss, _), grads = grad_fn(data, label)
                loss = ops.depend(loss, optimizer(grads))
                return loss

            size = dataset.get_dataset_size()
            model.set_train()
            for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
                loss = train_step(data, label)
                if batch == 0:
                    break
                    loss, current = loss.asnumpy(), batch
                    print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")

        def test_loop(model, dataset, loss_fn):
            num_batches = dataset.get_dataset_size()
            model.set_train(False)
            total, test_loss, correct = 0, 0, 0
            for data, label in dataset.create_tuple_iterator():
                pred = model(data)
                total += len(data)
                test_loss += loss_fn(pred, label).asnumpy()
                correct += (pred.argmax(1) == label).asnumpy().sum()
            test_loss /= num_batches
            correct /= total
            print(f"Test: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        loss_fn = nn.CrossEntropyLoss()
        optimizer = nn.SGD(model.trainable_params(), learning_rate=learning_rate)

        epochs = 1
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loop(model, train_dataset, loss_fn, optimizer)
            # test_loop(model, test_dataset, loss_fn)
        print("Done!")

    delete_file("/tmp/", file_name="mindspore_tracking_test.log")
    main()
    assert file_and_key_match("/tmp/", "logits = Tensor(shape=[64, 10], dtype=Float32, value=[[",
                              file_name="mindspore_tracking_test.log")


def test_sequentialcell_level1():
    @ts.tracking(level=1, output="/tmp/mindspore_tracking_test.log")
    def main():
        context.set_context(mode=context.PYNATIVE_MODE)
        conv = nn.Conv2d(3, 2, 3, pad_mode='valid', weight_init="ones")
        relu = nn.ReLU()
        seq = nn.SequentialCell([conv, relu])
        x = Tensor(np.ones([1, 3, 4, 4]), dtype=mindspore.float32)
        output = seq(x)
        print(output)

    delete_file("/tmp/", file_name="mindspore_tracking_test.log")
    main()
    assert file_and_key_match("/tmp/", "cell = Conv2d<input_channels=3, output_channels=2,",
                              file_name="mindspore_tracking_test.log")


def test_sequentialcell_level2():
    @ts.tracking(level=2, output="/tmp/mindspore_tracking_test.log")
    def main():
        context.set_context(mode=context.PYNATIVE_MODE)
        conv = nn.Conv2d(3, 2, 3, pad_mode='valid', weight_init="ones")
        relu = nn.ReLU()
        seq = nn.SequentialCell([conv, relu])
        x = Tensor(np.ones([1, 3, 4, 4]), dtype=mindspore.float32)
        output = seq(x)
        print(output)

    delete_file("/tmp/", file_name="mindspore_tracking_test.log")
    main()
    assert file_and_key_match("/tmp/", "def _convert_python_data(data):",
                              file_name="mindspore_tracking_test.log")


def test_sequentialcell_level3():
    @ts.tracking(level=3, output="/tmp/mindspore_tracking_test.log")
    def main():
        context.set_context(mode=context.PYNATIVE_MODE)
        conv = nn.Conv2d(3, 2, 3, pad_mode='valid', weight_init="ones")
        relu = nn.ReLU()
        seq = nn.SequentialCell([conv, relu])
        x = Tensor(np.ones([1, 3, 4, 4]), dtype=mindspore.float32)
        output = seq(x)
        print(output)

    delete_file("/tmp/", file_name="mindspore_tracking_test.log")
    main()
    assert file_and_key_match("/tmp/", "def _run_construct(self, cast_inputs, kwargs):",
                              file_name="mindspore_tracking_test.log")


def test_sequentialcell_level4():
    @ts.tracking(level=4, output="/tmp/mindspore_tracking_test.log")
    def main():
        context.set_context(mode=context.PYNATIVE_MODE)
        conv = nn.Conv2d(3, 2, 3, pad_mode='valid', weight_init="ones")
        relu = nn.ReLU()
        seq = nn.SequentialCell([conv, relu])
        x = Tensor(np.ones([1, 3, 4, 4]), dtype=mindspore.float32)
        output = seq(x)
        print(output)

    delete_file("/tmp/", file_name="mindspore_tracking_test.log")
    main()
    assert file_and_key_match("/tmp/", "codecs.py",
                              file_name="mindspore_tracking_test.log")


def test_sqrt_nan(caplog):
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.sqrt = ops.Sqrt()
            self.matmul = ops.MatMul()

        def construct(self, input_x):
            y = self.matmul(input_x, input_x)
            x = self.sqrt(y)
            return x

    # @ts.tracking(level=1, path_bl=['context.py'])
    @ts.tracking(level=2, check_keyword='nan', check_mode=2, color=False)
    def nan_func():
        input_x = Tensor(np.array([[0.0, -1.0], [4.0, 3.0]]))
        k = 3.0
        net = Net()
        print(net(input_x))

    nan_func()
    captured_list = caplog.messages
    count = 0
    for cap in captured_list:
        if cap.find("'User Warning 'NAN'") == -1:
            count = count + 1
    assert count == 2


def test_white_list(capsys):
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    @ts.tracking(level=3, color=False, path_wl=['layer/conv.py'])
    def main():
        context.set_context(mode=context.PYNATIVE_MODE)
        conv = nn.Conv2d(3, 2, 3, pad_mode='valid', weight_init="ones")
        relu = nn.ReLU()
        seq = nn.SequentialCell([conv, relu])
        x = Tensor(np.ones([1, 3, 4, 4]), dtype=mindspore.float32)
        output = seq(x)
        #print(output)

    main()
    result = capsys.readouterr().err
    assert result.count("Source path:") == 2


def test_func_white_list(capsys):
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    # path_bl=['layer/activation.py']
    @ts.tracking(level=2, color=False, path_bl=['layer/activation.py'])
    def main():
        context.set_context(mode=context.PYNATIVE_MODE)
        conv = nn.Conv2d(3, 2, 3, pad_mode='valid', weight_init="ones")
        relu = nn.ReLU()
        seq = nn.SequentialCell([conv, relu])
        x = Tensor(np.ones([1, 3, 4, 4]), dtype=mindspore.float32)
        output = seq(x)
        print(output)

    main()
    result = capsys.readouterr().err
    #print(result)
    assert result.count("activation.py") == 0


def test_func_white_list(capsys):
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    # path_bl=['layer/activation.py']
    @ts.tracking(level=2, color=False, path_bl=['layer/activation.py'])
    def main():
        context.set_context(mode=context.PYNATIVE_MODE)
        conv = nn.Conv2d(3, 2, 3, pad_mode='valid', weight_init="ones")
        relu = nn.ReLU()
        seq = nn.SequentialCell([conv, relu])
        x = Tensor(np.ones([1, 3, 4, 4]), dtype=mindspore.float32)
        output = seq(x)
        print(output)

    main()
    result = capsys.readouterr().err
    #print(result)
    assert result.count("activation.py") == 0


def test_func_event_list(capsys):
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    # path_bl=['layer/activation.py']
    @ts.tracking(level=2, color=False, event_list=['call'])
    def main():
        context.set_context(mode=context.PYNATIVE_MODE)
        conv = nn.Conv2d(3, 2, 3, pad_mode='valid', weight_init="ones")
        relu = nn.ReLU()
        seq = nn.SequentialCell([conv, relu])
        x = Tensor(np.ones([1, 3, 4, 4]), dtype=mindspore.float32)
        output = seq(x)
        print(output)

    main()
    result = capsys.readouterr().err
    #print(result)
    assert result.count(" line ") == 0