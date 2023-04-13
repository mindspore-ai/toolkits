#!/usr/bin/env python3
# coding=utf-8

"""
This is a python code template
"""

import os
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.vision.py_transforms as P
from mindspore.dataset.vision import Inter
from tests.util import delete_file, file_and_key_match
from mindspore.mindrecord import FileWriter

import random
from PIL import Image
import troubleshooter as ts
import numpy as np

imagenet_path = "../../data"


def test_dataset_batch():
    class GetDatasetGenerator:
        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.seg_name = os.listdir(os.path.join(root_dir, "n02095570"))
            self.img_name = os.listdir(os.path.join(root_dir, "n02095570"))

        def __getitem__(self, index):
            segment_name = self.seg_name[index]
            img_name = self.seg_name[index]
            segment_path = os.path.join(self.root_dir, "n02095570", segment_name)
            img_path = os.path.join(self.root_dir, "n02095570", img_name)
            image_img = Image.open(img_path)
            segment_img = Image.open(segment_path)

            image_np = np.array(image_img)
            segment_np = np.array(segment_img)

            return image_np, segment_np

        def __len__(self):
            return len(self.img_name)

    @ts.proposal(write_file_path="/tmp/")
    def test_batch_error():
        generator = GetDatasetGenerator(imagenet_path)
        dataset = ds.GeneratorDataset(generator, ["data", "label"], shuffle=False)

        resize_op = C.Resize(size=(388, 388), interpolation=Inter.BILINEAR)
        rescale_op = C.Rescale(1.0 / 127.5, -1)
        pad_op = C.Pad(padding=92)

        c_trans = [rescale_op, pad_op]
        dataset = dataset.map(input_columns="data", operations=c_trans)
        dataset = dataset.map(input_columns="label", operations=pad_op)
        dataset = dataset.batch(batch_size=12)
        for data in dataset.create_dict_iterator():
            print(data["data"].shape, data["label"].shape)

    delete_file("/tmp/")
    test_batch_error()
    assert file_and_key_match("/tmp/", "dataset_id_2")


def test_dataset_filewriter():
    data_record_path = '../../data/test.mindrecord'

    @ts.proposal(write_file_path="/tmp/")
    def main():
        writer = FileWriter(file_name=data_record_path, shard_num=4)

        # 定义schema
        data_schema = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}
        writer.add_schema(data_schema, "test_schema")

        # 数据准备
        file_name = "../../data/000000581721.jpg"
        with open(file_name, "rb") as f:
            bytes_data = f.read()
        data = [{"file_name": "transform.jpg", "label": 1, "data": bytes_data}]

        indexes = ["file_name", "label"]
        writer.add_index(indexes)

        # 数据写入
        writer.write_raw_data(data)

        # 生成本地数据
        writer.commit()

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "dataset_id_4")


def test_dataset_imagetype():
    class GetDatasetGenerator:
        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.seg_name = os.listdir(os.path.join(root_dir, "n02095570"))
            self.img_name = os.listdir(os.path.join(root_dir, "n02095570"))

        def __getitem__(self, index):
            segment_name = self.seg_name[index]
            img_name = self.seg_name[index]
            segment_path = os.path.join(self.root_dir, "n02095570", segment_name)
            img_path = os.path.join(self.root_dir, "n02095570", img_name)
            image_img = Image.open(img_path)
            segment_img = Image.open(segment_path)

            image_np = np.array(image_img)
            segment_np = np.array(segment_img)

            return image_np, segment_np

        def __len__(self):
            return len(self.img_name)

    @ts.proposal(write_file_path="/tmp/")
    def test_image_type_error():
        generator = GetDatasetGenerator(imagenet_path)
        dataset = ds.GeneratorDataset(generator, ["data", "label"], shuffle=False)

        resize_op = C.Resize(size=(388, 388), interpolation=Inter.BILINEAR)
        rescale_op = C.Rescale(1.0 / 127.5, -1)
        pad_op = C.Pad(padding=92)

        p_trans = [P.Resize((388, 388))]
        # p_trans = [P.ToPIL(), P.Resize((388,388)), P.ToTensor(output_type=np.float32)]
        dataset = dataset.map(input_columns="data", operations=p_trans)
        for data in dataset.create_dict_iterator():
            print(data["data"].shape, data["label"].shape)

    delete_file("/tmp/")
    test_image_type_error()
    assert file_and_key_match("/tmp/", "dataset_id_3")


def test_dataset_memory():
    @ts.proposal(write_file_path="/tmp/")
    def main():
        print("Test dataset memory not enough.")
        raise RuntimeError(f"Memory not enough: current free memory size")

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "dataset_id_14")


def test_dataset_pyfunc1():
    class GetDatasetGenerator:
        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.seg_name = os.listdir(os.path.join(root_dir, "n02095570"))
            self.img_name = os.listdir(os.path.join(root_dir, "n02095570"))

        def __getitem__(self, index):
            segment_name = self.seg_name[index]
            img_name = self.seg_name[index]
            segment_path = os.path.join(self.root_dir, "n02095570", segment_name)
            img_path = os.path.join(self.root_dir, "n02095570", img_name)
            image_img = Image.open(img_path)
            segment_img = Image.open(segment_path)

            image_np = np.array(image_img)
            segment_np = np.array(segment_img)

            return image_np, segment_np

        def __len__(self):
            return len(self.img_name)

    class pyfunc_input_error():
        def __init__(self):
            pass

        def __call__(self, sample):
            print(type(sample))
            data, label = sample['data'], sample['label']
            return data, lable

    @ts.proposal(write_file_path="/tmp/")
    def test_pyfunc_input_error():
        generator = GetDatasetGenerator(imagenet_path)
        dataset = ds.GeneratorDataset(generator, ["data", "label"], shuffle=False)

        dataset = dataset.map(operations=[pyfunc_input_error()])
        for data in dataset.create_dict_iterator():
            print(data["data"].shape, data["label"].shape)

    delete_file("/tmp/")
    test_pyfunc_input_error()
    assert file_and_key_match("/tmp/", "dataset_id_13")


def test_dataset_pyfunc2():
    class GetDatasetGenerator:
        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.seg_name = os.listdir(os.path.join(root_dir, "n02095570"))
            self.img_name = os.listdir(os.path.join(root_dir, "n02095570"))

        def __getitem__(self, index):
            segment_name = self.seg_name[index]
            img_name = self.seg_name[index]
            segment_path = os.path.join(self.root_dir, "n02095570", segment_name)
            img_path = os.path.join(self.root_dir, "n02095570", img_name)
            image_img = Image.open(img_path)
            segment_img = Image.open(segment_path)

            image_np = np.array(image_img)
            segment_np = np.array(segment_img)

            return image_np, segment_np

        def __len__(self):
            return len(self.img_name)

    class pyfunc_crop_error():
        def __init__(self, scale, patch):
            self.s = scale
            self.patch = patch

        def __call__(self, image, label):
            print(type(image))
            data, label = image, label
            h, w = data.shape[:2]

            ix = random.randrange(0, w - self.patch + 1)
            iy = random.randrange(0, h - self.patch + 1)

            data_patch = data[..., iy: iy + self.patch, ix: ix + self.patch]
            label_patch = label[..., iy: iy + self.patch, ix: ix + self.patch]

            return {'data': data_patch, 'label': label_patch}

    @ts.proposal(write_file_path="/tmp/")
    def test_pyfunc_crop_error():
        generator = GetDatasetGenerator(imagenet_path)
        dataset = ds.GeneratorDataset(generator, ["data", "label"], shuffle=False)

        resize_op = C.Resize(size=(388, 388))
        # c_trans = [resize_op, py_func_crop(2, 24)]
        c_trans = [resize_op]
        dataset = dataset.map(operations=c_trans, input_columns=["data"])
        dataset = dataset.map(operations=c_trans, input_columns=["label"])
        dataset = dataset.map(operations=[pyfunc_crop_error(2, 24)], input_columns=["data", "label"])

        for data in dataset.create_dict_iterator():
            print(data["data"].shape, data["label"].shape)

    delete_file("/tmp/")
    test_pyfunc_crop_error()
    assert file_and_key_match("/tmp/", "dataset_id_15")


def test_dataset_sampler():
    class MySampler(ds.Sampler):
        def __init__(self):
            # self.num_samples = 10
            self.dataset_size = 0
            self.child_sampler = None

        def __iter__(self):
            for i in range(0, 10, 2):
                yield i

    @ts.proposal(write_file_path="/tmp/")
    def main():
        DATA_DIR = "../../data/cifar-10-batches/"
        dataset = ds.Cifar10Dataset(DATA_DIR, sampler=MySampler())
        for data in dataset.create_dict_iterator():
            print("Image shape:", data['image'].shape, ", Label:", data['label'])

    delete_file("/tmp/")
    main()
    assert file_and_key_match("/tmp/", "dataset_id_7")
