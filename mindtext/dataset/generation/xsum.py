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
    XSUM class
"""
from typing import Union, List, Dict, Optional

import pandas as pd
from pandas import DataFrame
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from mindspore.mindrecord import FileWriter

from .. import Vocabulary
from ..base_dataset import Dataset


class XSUMDataset(Dataset):
    """
    QNLI dataset.

    Args:
        path (str, Optional): Dataset file path or Dataset directory path, default None.
        tokenizer (Union[str]): Tokenizer function, default 'spacy'.
        lang (str): Tokenizer language, default 'en'.
        max_size (int, Optional): Vocab max size, default None.
        min_freq (int, Optional): Min word frequency, default None.
        padding (str): Padding token, default `<pad>`.
        unknown (str): Unknown token, default `<unk>`.
        buckets (List[int], Optional): Padding row to the length of buckets, default None.

    Examples:
        >>> xsum = XSUMDataset(tokenizer='spacy', lang='en')
        # xsum = XSUMDataset(tokenizer='spacy', lang='en', buckets=[16,32,64])
        >>> ds = xsum()
    """

    def __init__(self, path: Optional[str] = None, tokenizer: Union[str] = 'spacy', lang: str = 'en',
                 max_size: Optional[int] = None, min_freq: Optional[int] = None, padding: str = '<pad>',
                 unknown: str = '<unk>', buckets: Optional[List[int]] = None, **kwargs):
        super(XSUMDataset, self).__init__(name='XSUM', **kwargs)
        if not isinstance(path, str):
            self._path = 'xsum'
        else:
            self._path = path
        self._tokenize = tokenizer
        self._lang = lang
        self._vocab_max_size = max_size
        self._vocab_min_freq = min_freq
        self._padding = padding
        self._unknown = unknown
        self._buckets = buckets
        self._stream_process = None

    def __call__(self):
        self.load(self._path)
        self.process(tokenizer=self._tokenize, lang=self._lang, max_size=self._vocab_max_size,
                     min_freq=self._vocab_min_freq, padding=self._padding,
                     unknown=self._unknown, buckets=self._buckets)
        return self.mind_datasets

    def _process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str,
                 unknown: str, dataset_type: str, buckets: List[int]) -> DataFrame:
        # Whether using a pretrained model tokenizer.
        if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
            dataset['document'] = self.tokenize_progress(dataset, dataset_type, 'document')
            dataset['summary'] = self.tokenize_progress(dataset, dataset_type, 'summary')

            if dataset_type == 'train':
                self._vocab = Vocabulary.from_dataset(dataset, field_name=['document', 'summary'], max_size=max_size,
                                                      min_freq=min_freq,
                                                      padding=padding, unknown=unknown)

            def _stream_process(row: Dict[str, pd.Series]):
                documnet_index = [self._vocab[i] for i in row['document']]
                summary_index = [self._vocab[i] for i in row['summary']]

                documnet_length = len(documnet_index)
                summary_length = len(summary_index)
                return documnet_index, documnet_length, summary_index, summary_length
        else:
            self._pretrained_model_inputs = self._tokenizer("").keys()

            def _stream_process(row: Dict[str, pd.Series]):
                result_dict_input = self._tokenizer(row['document'])
                result_dict_output = self._tokenizer.tokenize(row['summary'])
                result_dict_output = self._tokenizer.convert_tokens_to_ids(result_dict_output)
                result_dict_output = self._tokenizer.build_inputs_with_special_tokens(result_dict_output)
                return result_dict_input, result_dict_output
        self._stream_process = _stream_process
        return dataset

    def load(self, paths: Optional[str] = None) -> Dict[str, DataFrame]:
        self._datasets = load_dataset(self._path)
        self._datasets['dev'] = self._datasets.pop('validation')
        dict_dataset = {}
        for i in self._datasets.keys():
            pd_dataset = pd.DataFrame(columns=self._datasets[i].column_names)
            for j in self._datasets[i].column_names:
                pd_dataset[j] = pd.Series(self._datasets[i][j])
            dict_dataset[i] = pd_dataset
        self._datasets = dict_dataset
        return self._datasets

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
        # Whether using a pretrained model tokenizer.
        if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
            data_schema = {
                'input_ids': {'type': 'int64', 'shape': [-1]},
                'input_length': {'type': 'int64', 'shape': [-1]},
                'output_ids': {'type': 'int64', 'shape': [-1]},
                'output_length': {'type': 'int64', 'shape': [-1]}}
        else:
            data_schema = {}
            for i in self._pretrained_model_inputs:
                data_schema[i] = {'type': 'int64', 'shape': [-1]}
            data_schema['output_ids'] = {'type': 'int64', 'shape': [-1]}

        writer.add_schema(data_schema, self._name)
        data = []
        vocab_bar = tqdm(dataset.iterrows(), total=len(dataset))
        for index, row in vocab_bar:
            # Whether using a pretrained model tokenizer.
            if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
                input_ids, input_length, output_ids, output_length = self._stream_process(row)
                sample = {'input_ids': np.array(input_ids, dtype=np.int64),
                          'input_length': np.array(input_length, dtype=np.int64),
                          'output_ids': np.array(output_ids, dtype=np.int64),
                          'output_length': np.array(output_length, dtype=np.int64)}
            else:
                sample = {}
                input_dict, output = self._stream_process(row)
                for i in self._pretrained_model_inputs:
                    sample[i] = np.array(input_dict[i], dtype=np.int64)
                sample['output_ids'] = np.array(output, dtype=np.int64)
            data.append(sample)
            if index % 10 == 0:
                writer.write_raw_data(data)
                data = []
            vocab_bar.set_description("Writing data to .mindrecord file")
        if data:
            writer.write_raw_data(data)
        writer.commit()
        return list(data_schema.keys())

    def _load(self, path: str) -> DataFrame:
        pass
