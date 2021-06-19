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
"""
Data preprocess
"""
import os

from mindtext.classification.dataset.FastTextDataPreProcess import FastTextDataPreProcess
from mindtext.classification.utils import parse_args, get_config
from .mindrecord import mindrecord_file_path
from .load_data import load_dataset

DATASETS_TYPE = ['train', 'valid']


def create_dataset(config, select_dataset):
    """
    :param config:AttrDict
    :param select_dataset:a list or str, such as ['train','valid'] or 'train'
    :return:
    """
    # 增加返回train或者test数据集的条件判断
    if not isinstance(select_dataset, list) and not isinstance(select_dataset, str):
        raise ValueError("select_dataset must be a list or str")
    if isinstance(select_dataset, str):
        select_dataset = [select_dataset]
    for i in select_dataset:
        if i not in DATASETS_TYPE:
            raise ValueError("select_dataset should be either or both of {}.".format(DATASETS_TYPE))

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

    data_examples = {}
    for i in DATASETS_TYPE:
        data_examples[i] = None
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
        data_examples = {DATASETS_TYPE[0]: train_data_example, DATASETS_TYPE[1]: test_data_example}
        print("Data preprocess done")

    path = {}  # 将训练集与测试集写入mindrecord文件
    for i in DATASETS_TYPE:
        print("{} Data processing".format(i.title()))
        data_temp_dir = os.path.join(temp_dir, "{}_temp_data".format(i))
        data_path = mindrecord_file_path(config[i.upper()], data_temp_dir, data_examples[i])
        path[i] = data_path

    return_ = []  # 根据select_dataset参数返回对应的训练集或者测试集或者两者
    for i in select_dataset:
        if i != DATASETS_TYPE[0]:
            epoch_count = -1
        else:
            epoch_count = config[i.upper()].epoch_count
        return_.append(load_dataset(dataset_path=path[i],
                                    batch_size=config[i.upper()].batch_size,
                                    epoch_count=epoch_count,
                                    bucket=config[i.upper()].buckets))

    print("Create datasets done.....")
    if len(select_dataset) == 2:
        return return_[0], return_[1]
    if len(select_dataset) == 1:
        return return_[0]


if __name__ == '__main__':
    args = parse_args()
    config = get_config(args.config_path, overrides=args.override)
    create_dataset(config)
