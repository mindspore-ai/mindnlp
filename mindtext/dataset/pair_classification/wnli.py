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
# See the License for the sp    ecific language governing permissions and
# limitations under the License.
# ============================================================================
"""
    MRPC dataset
"""
from typing import Union, Dict, List

from tqdm import tqdm
import pandas as pd
from pandas import DataFrame

from ..base_dataset import PairCLSBaseDataset


class WNLIDataset(PairCLSBaseDataset):
    """
    WNLI dataset.

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
        >>> wnli = WNLIDataset(tokenizer='spacy', lang='en')
        # wnli = WNLIDataset(tokenizer='spacy', lang='en', buckets=[16,32,64])
        >>> ds = wnli()
    """

    def __init__(self, paths: Union[str, Dict[str, str]] = None,
                 tokenizer: Union[str] = 'spacy', lang: str = 'en', max_size: int = None, min_freq: int = None,
                 padding: str = '<pad>', unknown: str = '<unk>',
                 buckets: List[int] = None):
        super(WNLIDataset, self).__init__(sep='\t', name='WNLI', label_map={"0": 0, "1": 1})
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
            dataset = pd.read_csv(f, sep='\n', names=columns)
            tqdm.pandas(desc=f"{self._name} dataset loadding")
            if "label" in dataset.columns.values:
                dataset = dataset[['index', 'sentence1', 'sentence2', 'label']]
                dataset.columns = ['index', 'sentence1', 'sentence2', 'label']
                dataset[['index', 'sentence1', 'sentence2', 'label']] = dataset.progress_apply(_split_row, axis=1,
                                                                                               result_type="expand")
            else:
                dataset = dataset[['index', 'sentence1', 'sentence2']]
                dataset.columns = ['index', 'sentence1', 'sentence2']
                dataset[['index', 'sentence1', 'sentence2']] = dataset.progress_apply(_split_row, axis=1,
                                                                                      result_type="expand")
            dataset.fillna('')
            dataset.dropna(inplace=True)
        return dataset


def _split_row(row):
    row_data = row['index'].strip().split('\t')
    return row_data
