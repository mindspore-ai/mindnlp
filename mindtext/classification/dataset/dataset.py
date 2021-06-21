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

DATASETS_TYPE = ['train', 'valid', 'test']


def _data_preprocess(config, temp_dir, select_dataset):
    """
    Preprocessing according to the data_path provided by YAML configuration file
    """
    train_feature_dicts = {}
    for i in config.TRAIN.buckets:
        train_feature_dicts[i] = []
    valid_feature_dicts = {}
    for i in config.VALID.buckets:
        valid_feature_dicts[i] = []
    test_feature_dicts = {}
    for i in config.INFER.buckets:
        test_feature_dicts[i] = []

    data_examples = {}
    for i in DATASETS_TYPE:
        data_examples[i] = None
    _d = {}
    for i in DATASETS_TYPE:
        _d[i] = None
    feature_dicts = {'train': train_feature_dicts, 'valid': valid_feature_dicts, 'test': test_feature_dicts}
    _path = {'train': config.TRAIN.data_path.strip(), 'valid': config.VALID.data_path.strip(),
             'test': config.INFER.data_path.strip()}

    for i in select_dataset:
        if not os.path.exists(os.path.join(os.getcwd(), temp_dir, "{}_temp_data".format(i))):
            _d[i] = FastTextDataPreProcess(data_path=_path[i],
                                           max_length=config.PREPROCESS.max_len,
                                           ngram=2,
                                           class_num=config.MODEL_PARAMETERS.num_class,
                                           feature_dict=feature_dicts[i],
                                           buckets=config["INFER" if i == "test" else i.upper()].buckets,
                                           is_hashed=False,
                                           feature_size=10000000,
                                           is_train=True if i == "train" else False)

    for i in select_dataset:
        if _d[i] is not None:
            vocab_path = os.path.join(os.getcwd(), temp_dir, "vocab.txt")
            if i != "train" and os.path.exists(vocab_path):
                _d[i].read_vocab_txt(vocab_path)
            print("Begin to process {} data...".format(i))
            data_examples[i] = _d[i].load()
            print("{} data preprocess done".format(i))
            if i == "train":
                if not os.path.exists(os.path.join(os.getcwd(), temp_dir)):
                    os.makedirs(os.path.join(os.getcwd(), temp_dir))
                    _d[i].vocab_to_txt(vocab_path)
    return data_examples


def create_dataset(config, select_dataset):
    """
    :param config:AttrDict
    :param select_dataset:a list or str, such as ['train','valid'] or 'train'
    :return:
    """
    # SELECT_DATASET value and type checking
    if not isinstance(select_dataset, list) and not isinstance(select_dataset, str):
        raise ValueError("select_dataset must be a list or str")
    if isinstance(select_dataset, str):
        select_dataset = [select_dataset]
    for i in select_dataset:
        if i not in DATASETS_TYPE:
            raise ValueError("select_dataset should be one of {}.".format(DATASETS_TYPE))

    # If MID_DIR_PATH is not specified, the default path is "./mindrecord_data"
    default_temp_dir = 'mindrecord_data'
    if "mid_dir_path" not in config.PREPROCESS.keys():
        temp_dir = default_temp_dir
    else:
        temp_dir = config.PREPROCESS.mid_dir_path.strip()
    if len(temp_dir) == 0:
        temp_dir = default_temp_dir

    # Preprocessing dataset
    data_examples = _data_preprocess(config, temp_dir, select_dataset)

    # Write the dataset in mindrecord format under the MID_DIR_PATH
    path = {}
    for i in select_dataset:
        _config = config["INFER" if i == "test" else i.upper()]
        print("{} Data processing".format(i.title()))
        data_temp_dir = os.path.join(temp_dir, "{}_temp_data".format(i))
        data_path = mindrecord_file_path(_config, data_temp_dir, data_examples[i])
        path[i] = data_path

    # Read the dataset according to the SELECT_DATASET
    return_ = []
    for i in select_dataset:
        _config = config["INFER" if i == "test" else i.upper()]
        if i != DATASETS_TYPE[0]:
            epoch_count = -1
        else:
            epoch_count = _config.epoch_count
        return_.append(load_dataset(dataset_path=path[i],
                                    batch_size=_config.batch_size,
                                    epoch_count=epoch_count,
                                    bucket=_config.buckets))
    print("Create datasets done.....")
    if len(select_dataset) == 3:
        return return_[0], return_[1], return_[2]
    if len(select_dataset) == 2:
        return return_[0], return_[1]
    if len(select_dataset) == 1:
        return return_[0]


if __name__ == '__main__':
    args = parse_args()
    config = get_config(args.config_path, overrides=args.override)
    create_dataset(config)
