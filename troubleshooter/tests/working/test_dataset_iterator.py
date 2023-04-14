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

import random
from PIL import Image
import troubleshooter as ts
import numpy as np

imagenet_path = "../data"


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
def test_create_iterator():
    generator = GetDatasetGenerator(imagenet_path)
    dataset = ds.GeneratorDataset(generator, ["data", "label"], shuffle=False)

    resize_op = C.Resize(size=(388, 388), interpolation=Inter.BILINEAR)
    rescale_op = C.Rescale(1.0 / 127.5, -1)
    pad_op = C.Pad(padding=92)
    c_trans = [rescale_op, resize_op, pad_op]

    dataset = dataset.map(input_columns="data", operations=c_trans)
    for data in dataset.create_dict_iterator():
        print(data["data"].shape, data["label"].shape)


if __name__ == '__main__':
    test_create_iterator()
