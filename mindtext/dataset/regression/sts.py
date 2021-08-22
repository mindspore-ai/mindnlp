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
    STS-B dataset
"""
from typing import Union, Dict, List
import csv

import pandas as pd
from pandas import DataFrame

from ..base_dataset import PairCLSBaseDataset


class STSBDataset(PairCLSBaseDataset):
    """
    STS-B dataset.

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
        >>> sts = STSBDataset(tokenizer='spacy', lang='en')
        # stsb = STSBDataset(tokenizer='spacy', lang='en', buckets=[16,32,64])
        >>> ds = sts()
    """

    def __init__(self, paths: Union[str, Dict[str, str]] = None,
                 tokenizer: Union[str] = 'spacy', lang: str = 'en', max_size: int = None, min_freq: int = None,
                 padding: str = '<pad>', unknown: str = '<unk>',
                 buckets: List[int] = None):
        super(STSBDataset, self).__init__(sep='\t', name='STS-B', label_is_float=True)
        self._paths = paths
        self._tokenize = tokenizer
        self._lang = lang
        self._vocab_max_size = max_size
        self._vocab_min_freq = min_freq
        self._padding = padding
        self._unknown = unknown
        self._buckets = buckets

    def __call__(self):
        self.load(self._paths)
        self.process(tokenizer=self._tokenize, lang=self._lang, max_size=self._vocab_max_size,
                     min_freq=self._vocab_min_freq, padding=self._padding,
                     unknown=self._unknown, buckets=self._buckets)
        return self.mind_datasets

    def _load(self, path: str) -> DataFrame:
        with open(path, 'r', encoding='utf-8') as f:
            columns = f.readline().strip().split('\t')
            dataset = pd.read_csv(f, sep='\t', names=columns, quoting=csv.QUOTE_NONE)
            if "score" in dataset.columns.values:
                dataset = dataset[['index', 'sentence1', 'sentence2', 'score']]
                dataset.columns = ['index', 'sentence1', 'sentence2', 'label']
            else:
                dataset = dataset[['index', 'sentence1', 'sentence2']]
            dataset.fillna('')
            dataset.dropna(inplace=True)
        return dataset
