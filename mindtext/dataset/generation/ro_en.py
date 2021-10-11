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
    WMT16 Ro-En dataset
"""
from typing import Union, List, Optional

import pandas as pd
from pandas import DataFrame

from ..base_dataset import GenerateBaseDataset


class ROENDataset(GenerateBaseDataset):
    """
    Ro-En dataset.

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
        >>> roen = ROENDataset(tokenizer='facebook/bart')
        # roen = ROENDataset(tokenizer='facebook/bart')
        >>> ds = roen()
    """

    def __init__(self, paths: Optional[str] = None, tokenizer: Union[str] = 'spacy', lang: str = 'en',
                 max_size: Optional[int] = None, min_freq: Optional[int] = None, padding: str = '<pad>',
                 unknown: str = '<unk>', **kwargs):
        super(ROENDataset, self).__init__(name='ROEN', **kwargs)
        self._paths = paths
        self._tokenize = tokenizer
        self._lang = lang
        self._vocab_max_size = max_size
        self._vocab_min_freq = min_freq
        self._padding = padding
        self._unknown = unknown
        if self._stream:
            if (not isinstance(self._max_length, int) or not isinstance(self._max_pair_length, int)) and not isinstance(
                    self._buckets, List):
                raise TypeError(
                    "`max_length`, `max_pair_length` or `buckets` should be assigned when `stream` is `True`.")
        if bool(self._truncation_strategy) and not (
                isinstance(self._max_length, int) or isinstance(self._max_pair_length, int)) and not isinstance(
                    self._buckets, List):
            raise TypeError(
                "`truncation_strategy` need be `False` when `max_length` or `max_pair_length` is not assigned.")

    def __call__(self):
        self.load(self._paths)
        self.process(tokenizer=self._tokenize, lang=self._lang, max_size=self._vocab_max_size,
                     min_freq=self._vocab_min_freq, padding=self._padding,
                     unknown=self._unknown, buckets=self._buckets)
        return self._mind_datasets

    def _load(self, path: str) -> DataFrame:
        dataset = pd.read_csv(path, keep_default_na=False)
        dataset.columns = ['summary', 'document']
        return dataset
