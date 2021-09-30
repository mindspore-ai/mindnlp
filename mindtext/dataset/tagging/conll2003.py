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
"""conll2003 class"""
from typing import Union, Dict, List
import csv

import pandas as pd
from pandas import DataFrame

from mindtext.dataset.base_dataset import Dataset
import mindspore.dataset as ds

class CONLLDataset(Dataset):
    """
    CONLL Dataset.

    Args:
        paths (Union[str, Dict[str, str]], Optional): Dataset file path or Dataset directory path, default None.
        tokenizer (Union[str]): Tokenizer function, default 'spacy'.
        lang (str): Tokenizer language, default 'en'.
        max_size (int, Optional): Vocab max size, default None.
        min_freq (int, Optional): Min word frequency, default None.
        padding (str): Padding token, default `<pad>`.
        unknown (str): Unknown token, default `<unk>`.
        buckets (List[int], Optional): Padding row to the length of buckets, default None.

    Examples:
       #>>> conll = CONLLDataset(tokenizer='eng.train')
         # conll = CONLLDataset(tokenizer='eng.train')
       #>>> data = conll()
    """

    def __init__(self, paths: Union[str, Dict[str, str]] = None,
                 tokenizer: Union[str] = 'spacy', lang: str = 'en', max_size: int = None,
                 min_freq: int = None,
                 padding: str = '<pad>', unknown: str = '<unk>',
                 buckets: List[int] = None):
        super(CONLLDataset, self).__init__(sep='\t', name='conll')
        self._paths = paths
        self._tokenize = tokenizer
        self._lang = lang
        self._vocab_max_size = max_size
        self._vocab_min_freq = min_freq
        self._padding = padding
        self._unknown = unknown
        self._buckets = buckets

    def __call__(self) -> Dict[str, ds.MindDataset]:
        self.load(self._paths)
        self.process(tokenizer=self._tokenize, lang=self._lang, max_size=self._vocab_max_size,
                     min_freq=self._vocab_min_freq, padding=self._padding,
                     unknown=self._unknown, buckets=self._buckets)
        return self.mind_datasets

    def _load(self, path) -> DataFrame:
        """
                Load dataset from Conll file.

                Args:
                    path (str): Dataset file path.

                Returns:
                    DataFrame: Dataset file will be read as a DataFrame.
        """
        with open(path, "r", encoding='utf-8') as f:
            dataset = pd.read_csv(f, sep=' ', quoting=csv.QUOTE_NONE)
        print(dataset, '\n')
        return dataset

    def _process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                 dataset_type: str) -> DataFrame:
        """Preprocess dataset."""
        return None

    def _write_to_mr(self, dataset: DataFrame, file_path: Union[str, Dict[int, str]], is_test: bool,
                     process_function: callable = None) -> List[str]:
        """Write to .mindrecord file."""
        return None

    def _stream_process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                        dataset_type: str) -> callable:
        """Preprocess dataset by data stream."""
        return None
