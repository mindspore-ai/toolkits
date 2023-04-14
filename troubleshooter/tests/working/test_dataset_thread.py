# Copyright 2022 Tiger Miao and collaborators.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Data operations, will be used in train.py."""

import mindspore.common.dtype as mstype
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as deC

de.config.set_num_parallel_workers(32)


def create_transformer_dataset(rank_size=1, rank_id=0, do_shuffle="true", dataset_path=None, batch_size=16,
                               bucket_boundaries=None, device_target="Ascend"):
    """create dataset"""

    def batch_per_bucket(bucket_len, dataset_path):
        dataset_path = dataset_path + "_" + str(bucket_len) + "_00"
        ds = de.MindDataset(dataset_path,
                            columns_list=["source_eos_ids", "source_eos_mask",
                                          "target_sos_ids", "target_sos_mask",
                                          "target_eos_ids", "target_eos_mask"],
                            shuffle=(do_shuffle == "true"), num_shards=rank_size, shard_id=rank_id)
        type_cast_op = deC.TypeCast(mstype.int32)
        ds = ds.map(operations=type_cast_op, input_columns="source_eos_ids")
        ds = ds.map(operations=type_cast_op, input_columns="source_eos_mask")
        ds = ds.map(operations=type_cast_op, input_columns="target_sos_ids")
        ds = ds.map(operations=type_cast_op, input_columns="target_sos_mask")
        ds = ds.map(operations=type_cast_op, input_columns="target_eos_ids")
        ds = ds.map(operations=type_cast_op, input_columns="target_eos_mask")

        # apply batch operations

        ds = ds.batch(batch_size, drop_remainder=True)
        return ds

    for i, _ in enumerate(bucket_boundaries):
        bucket_len = bucket_boundaries[i]
        ds_per = batch_per_bucket(bucket_len, dataset_path)
        if i == 0:
            ds = ds_per
        else:
            ds = ds + ds_per
    ds = ds.shuffle(ds.get_dataset_size())
    ds.channel_name = 'transformer'
    return ds


def load_dataset(path):
    # dataset_path=config.data_path,
    bucket_boundaries = [16, 32, 48, 64, 128]
    dataset = create_transformer_dataset(dataset_path=path, bucket_boundaries=bucket_boundaries)
    dataset_size = dataset.get_dataset_size()
    print("dataset size: ", dataset_size)
    iter = dataset.create_dict_iterator()
    data = next(iter)
    print('len:', len(data))
    source_ids = data['source_eos_ids']
    source_mask = data['source_eos_mask']
    print("ids: ", source_ids.shape)
    print("mask: ", source_mask.shape)


if __name__ == "__main__":
    dataset_path = "/opt/nvme0n1/dataset/wmtende/data/transformer_mti/ende-l128-mindrecord"
    load_dataset(dataset_path)
