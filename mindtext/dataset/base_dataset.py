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
    Base Dataset
"""
import logging
from typing import Union, Dict, List
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter

from .utils import get_cache_path, _get_dataset_url, cached_path, check_loader_paths, get_tokenizer, \
    _preprocess_sequentially, _get_dataset_type
from . import Vocabulary, Pad

logging.basicConfig(level=logging.NOTSET)


class Dataset:
    """
    Base class of Dataset.

    Dataset supports the following five functions.
        - download(): Download dataset to default path:default path: ~/.mindtext/datasets/.
                the function will return the cache path of the downloaded file.
        - _load(): Read data from a data file, return :class:`panadas.DataFrame`.
        - load(): The files are read separately as DataFrame and are put into a dict.
        - _process(): Preprocess dataset.
        - process(): Preprocess dataset by a tokenizer.

    Args:
        vocab (Vocabulary, Optional): Convert tokens to index,default None.
        name (str, Optional): Dataset name,default None.
        label_map (Dict[str, int], Optional): Dataset label map,default None.
    """

    def __init__(self, vocab: Vocabulary = None, name: str = None,
                 label_map: Dict[str, int] = None):
        self._vocab = vocab
        self._name = name
        self._label_map = label_map
        self._datasets = None
        self._tokenizer = None
        self._mind_datasets = {}

    def _load(self, path: str) -> DataFrame:
        """
        Given a path, return the DataFrame.

        Args:
            path (str): Dataset file path.

        Returns:
            DataFrame: Dataset file will be read as a DataFrame.
        """
        raise NotImplementedError

    def load(self, paths: Union[str, Dict[str, str]] = None) -> Dict[str, DataFrame]:
        """
        Read data from a file in one or more specified paths.

        Args:
            paths (Union[str, Dict[str, str]], Optional): Dataset path, default None.

        Returns:
            Dict[str, DataFrame]: A Dataset dictionary.

        Examples::
            There are several inputs mode:
            0.If None, it checks to see if there is a local cache. If not, it is automatically downloaded and cached.
            1.given a directory, the "train" in directory will be considered to be train dataset::
                ds = xxxDataset().load('/path/dir')
                #  ds = {"train":..., "dev":..., "test":...} if the directory contains "train", "dev", "test".
            2.given a dict,such as train,dev,test not in the same directory,or the train, dev,
            test are not contained in directory::
                paths = {'train':"/path/to/train.tsv", 'dev':"/to/validate.tsv", "test":"/to/test.tsv"}
                ds = xxxDataset().load(paths)
                #  ds = {"train":..., "dev":..., "test":...}
            3.give a file name::
                ds = xxxDataset().load("/path/to/a/train.conll")
                tr_data = data_bundle.get_dataset('train')  # the file name contain "train".
        """
        if not paths:
            paths = self.download()
        paths = check_loader_paths(paths)
        self._datasets = {name: self._load(path) for name, path in paths.items()}
        return self._datasets

    def _process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                 dataset_type: str, buckets: List[int]) -> DataFrame:
        """
        Preprocess dataset.

        Args:
            dataset (DataFrame): DataFrame need to preprocess.
            max_size (int): Vocab max size.
            min_freq (int): Min word frequency.
            padding (str): Padding token.
            unknown (str): Unknown token.
            dataset_type (str): Dataset type(train, dev, test).
            buckets (List[int]): Padding row to the length of buckets.

        Returns:
            DataFrame: Preprocessed dataset.
        """
        raise NotImplementedError(f"{self.__class__} cannot be preprocessed")

    def process(self, tokenizer: Union[str], lang: str, max_size: int = None, min_freq: int = None,
                padding: str = '<pad>', unknown: str = '<unk>', buckets: List[int] = None) -> Dict[str, ds.MindDataset]:
        """
        Preprocess dataset.

        Args:
            tokenizer (Union[str]): Tokenizer function.
            lang (str): Tokenizer language.
            max_size (int, Optional): Vocab max size, default None.
            min_freq (int, Optional): Min word frequency, default None.
            padding (str): Padding token,default `<pad>`.
            unknown (str): Unknown token,default `<unk>`.
            buckets (List[int], Optional): Padding row to the length of buckets, default None.

        Returns:
            Dict[str, MindDataset]: A MindDataset dictionary.
        """
        if isinstance(tokenizer, str):
            self._tokenizer = get_tokenizer(tokenizer, lang=lang)

            dataset_file_name = _preprocess_sequentially(list(self._datasets.keys()))

            for dataset_name in dataset_file_name:
                dataset = self._datasets.get(dataset_name)
                d_t = _get_dataset_type(dataset_name)
                if isinstance(dataset, DataFrame):
                    dataset = self._process(dataset, max_size, min_freq, padding, unknown,
                                            dataset_type=d_t,
                                            buckets=buckets)
                    self._datasets[dataset_name] = dataset
                    dataset = self.convert_to_mr(dataset, buckets, dataset_name, is_test=d_t == 'test')
                    self._mind_datasets[dataset_name] = dataset
        return self._mind_datasets

    def convert_to_mr(self, dataset: DataFrame, buckets: List[int], file_name: str, is_test: bool) -> ds.MindDataset:
        """
        Convert dataset to .mindrecord format file,and read as MindDataset.

        Args:
            dataset (DataFrame): Tokenizer function.
            buckets (List[str]): Buckets of dataset.
            file_name (str): Name of .mindrecord file.
            is_test (bool): Whether the data set is a test set.

        Returns:
            MindDataset: A MindDataset.
        """
        mr_dir_path = Path(get_cache_path()) / Path("dataset") / Path(self._name).joinpath("mindrecord", file_name)
        if not mr_dir_path.exists():
            mr_dir_path.mkdir(parents=True, exist_ok=True)
        else:
            shutil.rmtree(mr_dir_path)
            mr_dir_path.mkdir(parents=True, exist_ok=True)
        md_dataset = None
        if not buckets:
            file_path = mr_dir_path.joinpath(file_name + ".mindrecord")
            field_name = self._write_to_mr(dataset, str(file_path), is_test)
            md_dataset = ds.MindDataset(dataset_file=str(file_path), columns_list=field_name)
        else:
            for i in range(len(buckets)):
                file_path = mr_dir_path.joinpath(file_name + "_" + str(buckets[i]) + ".mindrecord")
                if i == len(buckets) - 1:
                    dataset_bucket = dataset[dataset['padding_length'] >= buckets[i]]
                else:
                    dataset_bucket = dataset[dataset['padding_length'] == buckets[i]]
                if not dataset_bucket.index.empty:
                    field_name = self._write_to_mr(dataset_bucket, str(file_path), is_test)
                    if not md_dataset:
                        md_dataset = ds.MindDataset(dataset_file=str(file_path), columns_list=field_name)
                    else:
                        md_dataset += ds.MindDataset(dataset_file=str(file_path), columns_list=field_name)
        return md_dataset

    def _write_to_mr(self, dataset: DataFrame, file_path: str, is_test: bool) -> List[str]:
        """
        Write to .mindrecord file.

        Args:
            dataset (DataFrame): Tokenizer function.
            file_path (str): Path of mindrecord file.
            is_test (bool): Whether the data set is a test set.

        Returns:
            List[str]: Dataset field.
        """
        raise NotImplementedError

    @staticmethod
    def _get_dataset_path(dataset_name: str) -> Union[str, Path]:
        """
        Given a dataset name, try to read the dataset directory, if not exits,
        the function will try to download the corresponding dataset.

        Args:
            dataset_name (str): Dataset name.

        Returns:
             Union[str, Path]: Dataset directory path.
        """
        default_cache_path = get_cache_path()
        url = _get_dataset_url(dataset_name)
        output_dir = cached_path(url_or_filename=url, cache_dir=default_cache_path, name='dataset')

        return output_dir

    @property
    def vocab(self) -> Vocabulary:
        """
        Return vocabulary.

        Returns:
            Vocabulary: Dataset Vocabulary.
        """
        return self._vocab

    @property
    def datasets(self) -> Dict[str, DataFrame]:
        """
        Return datasets.

        Returns:
            Dict[str, DataFrame]: A dict of dataset.
        """
        return self._datasets

    @property
    def mind_datasets(self) -> Dict[str, ds.MindDataset]:
        """
        Return mindDataset.

        Returns:
            Dict[str, MindDataset]: A dict of mindDataset.
        """
        return self._mind_datasets

    def label_to_idx(self, row: str) -> int:
        """
        Convert label from a token to index.

        Args:
            row (str): Label tokens.

        Returns:
            str: Label index.
        """
        return self._label_map[row]

    def tokenize_progress(self, dataset: Union[DataFrame, pd.Series], dataset_type: str, field: str) \
            -> Union[DataFrame, pd.Series]:
        """
        Tokenizer with progress bar.

        Args:
            dataset (Union[DataFrame, Series]): Data need to be tokenized.
            dataset_type (str): Dataset type(train, dev, test).
            field (str): Field name.

        Returns:
            Union[DataFrame, Series]: Tokenized data.
        """
        tqdm.pandas(desc=f"{self._name} {dataset_type} dataset {field} preprocess bar(tokenize).")
        return dataset[field].progress_apply(self._tokenizer)

    def get_length_progress(self, dataset: Union[DataFrame, pd.Series], dataset_type: str, field: str) -> Union[
            DataFrame, pd.Series]:
        """
        Get sentence length.

        Args:
            dataset (Union[DataFrame, Series]): Data need to be processed.
            dataset_type (str): Dataset type(train, dev, test).
            field (str): Field name.

        Returns:
            Union[DataFrame, Series]: Processed data.
        """
        tqdm.pandas(desc=f"{self._name} {dataset_type} dataset {field} preprocess bar(length).")
        return dataset[field].progress_apply(len)

    def padding_progress(self, dataset: Union[DataFrame, pd.Series], dataset_type: str, field: str,
                         pad_function: Pad) -> Union[DataFrame, pd.Series]:
        """
        Padding index sequence.

        Args:
            dataset (Union[DataFrame, Series]): Data need to padding.
            dataset_type (str): Dataset type(train, dev, test).
            field (str): Field name.
            pad_function (Pad): Pad class.

        Returns:
            Union[DataFrame, Series]: Processed data.
        """
        tqdm.pandas(desc=f"{self._name} {dataset_type} dataset {field} preprocess bar(padding).")
        return dataset[field].progress_apply(pad_function)

    def padding_same_progress(self, dataset: Union[DataFrame, pd.Series], dataset_type: str,
                              field: Union[str, List[str]]) -> Union[DataFrame, pd.Series]:
        """
        Pad both sentences to the same length.

        Args:
            dataset (Union[DataFrame, Series]): Data need to padding.
            dataset_type (str): Dataset type(train, dev, test).
            field (Union[str, List[str]]): Field name.

        Returns:
            Union[DataFrame, Series]: Processed data.
        """

        def padding_same(row):
            if len(row['input1_ids']) > len(row['input2_ids']):
                input2 = Pad.padding(row['input2_ids'], len(row['input1_ids']), self._vocab.padding_idx)
                re = row['input1_ids'], input2
            else:
                input1 = Pad.padding(row['input1_ids'], len(row['input2_ids']), self._vocab.padding_idx)
                re = input1, row['input2_ids']
            return re

        tqdm.pandas(desc=f"{self._name} {dataset_type} dataset {field} preprocess bar(same padding).")
        dataset = dataset[field].progress_apply(padding_same, axis=1,
                                                result_type="expand")
        dataset.columns = field
        return dataset

    def download(self) -> str:
        """
        Dataset download.

        Returns:
            str: The downloaded dataset directory.
        """
        output_dir = self._get_dataset_path(dataset_name=self._name)
        return output_dir

    def __getitem__(self, dataset_type: str) -> DataFrame:
        """
        Return dataset by dataset_type.

        Args:
            dataset_type (str): Dataset type.

        Returns:
            DataFrame: Dataset(train, dev, test).
        """
        return self._datasets[dataset_type]

    def __str__(self) -> str:
        return str(dict(zip(self._datasets.keys(), [value.shape for value in self._datasets.values()])))


class CLSBaseDataset(Dataset):
    """
    A base class of text classification.

    Args:
        sep (str): The separator for pandas reading file, default ','.
    """

    def __init__(self, sep: str = ',', **kwargs):
        super(CLSBaseDataset, self).__init__(**kwargs)
        self.sep = sep
        self._label_nums = None

    def _load(self, path: str) -> DataFrame:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = pd.read_csv(f, sep=self.sep)
        dataset.fillna('')
        dataset.dropna(inplace=True)
        return dataset

    def _process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                 dataset_type: str, buckets: List[int]) -> DataFrame:
        """
        Classification dataset preprocess function.

        Args:
            dataset (DataFrame): DataFrame need to preprocess.
            max_size (int): Vocab max size.
            min_freq (int): Min word frequency.
            padding (str): Padding token.
            unknown (str): Unknown token.
            dataset_type (str): Dataset type(train, dev, test).
                Different types of datasets may be processed differently.
            buckets (List[int]): Padding row to the length of buckets.

        Returns:
            DataFrame: Preprocessed dataset.
        """
        dataset['sentence'] = self.tokenize_progress(dataset, dataset_type, 'sentence')

        if dataset_type == 'train':
            self._label_nums = dataset['label'].value_counts().shape[0]
            self._vocab = Vocabulary.from_dataset(dataset, field_name='sentence', max_size=max_size, min_freq=min_freq,
                                                  padding=padding, unknown=unknown)
        dataset['input_ids'] = self._vocab.word_to_idx(dataset['sentence'])
        dataset['input_length'] = self.get_length_progress(dataset, dataset_type, 'input_ids')
        if not buckets:
            max_length = dataset['input_length'].max()
            pad = Pad(max_length, self._vocab.padding_idx)
        else:
            pad = Pad(self._vocab.padding_idx, buckets=buckets)
        dataset['input_ids'] = self.padding_progress(dataset, dataset_type, 'input_ids', pad)
        dataset['padding_length'] = self.get_length_progress(dataset, dataset_type, 'input_ids')
        return dataset

    def _write_to_mr(self, dataset: DataFrame, file_path: str, is_test: bool) -> List[str]:
        """
        Write CLSDataset to .mindrecord file.

        Args:
            dataset (DataFrame): Tokenizer function.
            file_path (str): Path of mindrecord file.
            is_test (bool): Whether the data set is a test set.

        Returns:
            List[str]: Dataset field
        """
        writer = FileWriter(file_name=file_path, shard_num=1)
        data_schema = {
            'input_ids': {'type': 'int64', 'shape': [-1]},
            'input_length': {'type': 'int64', 'shape': [-1]}}
        if not is_test:
            data_schema['label'] = {'type': 'int64', 'shape': [-1]}
        writer.add_schema(data_schema, self._name)
        data = []
        vocab_bar = tqdm(dataset.iterrows(), total=len(dataset))
        for index, row in vocab_bar:
            sample = {'input_ids': np.array(row['input_ids'], dtype=np.int64),
                      'input_length': np.array(row['input_length'], dtype=np.int64)}
            if not is_test:
                sample['label'] = np.array(row['label'], dtype=np.int64)
            data.append(sample)
            if index % 10 == 0:
                writer.write_raw_data(data)
                data = []
            vocab_bar.set_description("Writing data to .mindrecord file")
        if data:
            writer.write_raw_data(data)
        writer.commit()
        return list(data_schema.keys())

    @property
    def label_nums(self) -> int:
        """
        Return label_nums.
        """
        return self._label_nums

    @label_nums.setter
    def label_nums(self, nums: int):
        """
        Need to be assigned.

        Args:
            nums (str): The number of label.
        """
        self._label_nums = nums


class PairCLSBaseDataset(Dataset):
    """
    A base class of  pair text classification.

    Args:
        sep (str): The separator for pandas reading file, default ','.
        label_is_float (bool): Whether the label of the dataset is float, default False.
    """

    def __init__(self, sep: str = ',', label_is_float: bool = False, **kwargs):
        super(PairCLSBaseDataset, self).__init__(**kwargs)
        self.sep = sep
        self._label_is_float = label_is_float
        self._label_nums = None

    def _load(self, path: str) -> DataFrame:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = pd.read_csv(f, sep=self.sep)
        dataset.fillna('')
        dataset.dropna(inplace=True)
        return dataset

    def _process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                 dataset_type: str, buckets: List[int]) -> DataFrame:
        """
        Pair text classification dataset preprocess function.

        Args:
            dataset (DataFrame): DataFrame need to preprocess.
            max_size (int): Vocab max size.
            min_freq (int): Min word frequency.
            padding (str): Padding token.
            unknown (str): Unknown token.
            dataset_type (str): Dataset type(train, dev, test).
                Different types of datasets may be preprocessed differently.
            buckets (List[int]): Padding row to the length of buckets.

        Returns:
            DataFrame: Preprocessed dataset.
        """
        dataset["sentence1"] = self.tokenize_progress(dataset, dataset_type, 'sentence1')
        dataset["sentence2"] = self.tokenize_progress(dataset, dataset_type, 'sentence2')

        if dataset_type != 'test':
            if not self._label_is_float and isinstance(self._label_map, Dict):
                dataset['label'] = dataset['label'].map(self.label_to_idx)

        if dataset_type == 'train':
            self._label_nums = dataset['label'].value_counts().shape[0]
            self._vocab = Vocabulary.from_dataset(dataset, field_name=["sentence1", "sentence2"], max_size=max_size,
                                                  min_freq=min_freq, padding=padding, unknown=unknown)
        dataset['input1_ids'] = self._vocab.word_to_idx(dataset['sentence1'])
        dataset['input2_ids'] = self._vocab.word_to_idx(dataset['sentence2'])
        dataset['input1_length'] = self.get_length_progress(dataset, dataset_type, 'input1_ids')
        dataset['input2_length'] = self.get_length_progress(dataset, dataset_type, 'input2_ids')
        if not buckets:
            max_length1 = dataset['input1_length'].max()
            max_length2 = dataset['input2_length'].max()
            pad1 = Pad(max_length1, self._vocab.padding_idx)
            pad2 = Pad(max_length2, self._vocab.padding_idx)
            dataset['input1_ids'] = self.padding_progress(dataset, dataset_type, 'input1_ids', pad1)
            dataset['input2_ids'] = self.padding_progress(dataset, dataset_type, 'input2_ids', pad2)
        else:
            pad = Pad(self._vocab.padding_idx, buckets=buckets)
            dataset['input1_ids'] = self.padding_progress(dataset, dataset_type, 'input1_ids', pad)
            dataset['input2_ids'] = self.padding_progress(dataset, dataset_type, 'input2_ids', pad)

        dataset[['input1_ids', 'input2_ids']] = self.padding_same_progress(dataset, dataset_type,
                                                                           ['input1_ids', 'input2_ids'])
        dataset['padding_length'] = self.get_length_progress(dataset, dataset_type, 'input1_ids')
        return dataset

    def _write_to_mr(self, dataset: DataFrame, file_path: str, is_test: bool) -> List[str]:
        """
        Write Pair text classification dataset to .mindrecord file.

        Args:
            dataset (DataFrame): Tokenizer function.
            file_path (str): Path of mindrecord file.
            is_test (bool): Whether the data set is a test set.

        Returns:
            List[str]: Dataset field.
        """
        writer = FileWriter(file_name=file_path, shard_num=1)
        data_schema = {
            'input1_ids': {'type': 'int64', 'shape': [-1]},
            'input1_length': {'type': 'int64', 'shape': [-1]},
            'input2_ids': {'type': 'int64', 'shape': [-1]},
            'input2_length': {'type': 'int64', 'shape': [-1]}}
        if not is_test:
            if not self._label_is_float:
                data_schema['label'] = {'type': 'int64', 'shape': [-1]}
            else:
                data_schema['label'] = {'type': 'float64', 'shape': [-1]}
        writer.add_schema(data_schema, self._name)
        data = []
        vocab_bar = tqdm(dataset.iterrows(), total=len(dataset))
        for index, row in vocab_bar:
            sample = {'input1_ids': np.array(row['input1_ids'], dtype=np.int64),
                      'input1_length': np.array(row['input1_length'], dtype=np.int64),
                      'input2_ids': np.array(row['input2_ids'], dtype=np.int64),
                      'input2_length': np.array(row['input2_length'], dtype=np.int64)}
            if not is_test:
                if not self._label_is_float:
                    sample['label'] = np.array(row['label'], dtype=np.int64)
                else:
                    sample['label'] = np.array(row['label'], dtype=np.float64)
            data.append(sample)
            if index % 10 == 0:
                writer.write_raw_data(data)
                data = []
            vocab_bar.set_description("Writing data to .mindrecord file")
        if data:
            writer.write_raw_data(data)
        writer.commit()
        return list(data_schema.keys())

    @property
    def label_nums(self) -> int:
        """
        Return label_nums.
        """
        return self._label_nums

    @label_nums.setter
    def label_nums(self, nums: int):
        """
        Need to be assigned.

        Args:
            nums (str): The number of label.
        """
        self._label_nums = nums
