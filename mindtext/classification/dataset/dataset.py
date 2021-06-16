# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Data preprocess"""
import os

from mindtext.classification.dataset.FastTextDataPreProcess import FastTextDataPreProcess
from mindtext.classification.utils import parse_args, get_config
from .mindrecord import mindrecord_file_path
from .load_data import load_dataset


def create_dataset(config):
    '''
    :param config:
    :return:
    '''

    train_feature_dicts = {}
    for i in config.TRAIN.buckets:
        train_feature_dicts[i] = []
    test_feature_dicts = {}
    for i in config.VALID.buckets:
        test_feature_dicts[i] = []

    default_temp_dir = 'mindrecord_data'
    if "mid_dir_path" not in config.PREPROCESS.keys():
        temp_dir = default_temp_dir
    else:
        temp_dir = config.PREPROCESS.mid_dir_path
    if len(temp_dir) == 0:
        temp_dir = default_temp_dir

    if not os.path.exists(os.path.join(os.getcwd(), temp_dir)):
        g_d = FastTextDataPreProcess(train_path=config.TRAIN.data_path,
                                     test_file=config.VALID.data_path,
                                     max_length=config.PREPROCESS.max_len,
                                     ngram=2,
                                     class_num=config.MODEL_PARAMETERS.num_class,
                                     train_feature_dict=train_feature_dicts,
                                     buckets=config.TRAIN.buckets,
                                     test_feature_dict=test_feature_dicts,
                                     test_bucket=config.VALID.buckets,
                                     is_hashed=False,
                                     feature_size=10000000)
        train_data_example, test_data_example = g_d.load()
        print("Data preprocess done")

    train_temp_dir = os.path.join(temp_dir, "train_temp_data")
    data_path = mindrecord_file_path(config.TRAIN, train_temp_dir, train_data_example)

    valid_temp_dir = os.path.join(temp_dir, "valid_temp_data")
    mindrecord_file_path(config.VALID, valid_temp_dir, test_data_example)

    print("All done.....")

    return load_dataset(dataset_path=data_path,
                        batch_size=config.TRAIN.batch_size,
                        epoch_count=config.TRAIN.epoch_count,
                        bucket=config.TRAIN.buckets)


if __name__ == '__main__':
    args = parse_args()
    config = get_config(args.config_path, overrides=args.override)
    create_dataset(config)
