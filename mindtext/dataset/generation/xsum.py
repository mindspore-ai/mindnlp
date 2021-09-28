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

from .. import Vocabulary, Pad
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

    def __init__(self, paths: Optional[str] = None, tokenizer: Union[str] = 'spacy', lang: str = 'en',
                 max_size: Optional[int] = None, min_freq: Optional[int] = None, padding: str = '<pad>',
                 unknown: str = '<unk>', **kwargs):
        super(XSUMDataset, self).__init__(name='XSUM', **kwargs)
        if not isinstance(paths, str):
            self._paths = 'xsum'
        else:
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

    def _stream_process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                        dataset_type: str) -> callable:
        """
        Preprocess dataset by data stream.

        Args:
            dataset (DataFrame): DataFrame need to preprocess.
            max_size (int): Vocab max size.
            min_freq (int): Min word frequency.
            padding (str): Padding token.
            unknown (str): Unknown token.
            dataset_type (str):  Dataset type(train, dev, test).
                Different types of datasets may be processed differently.

        Returns:
            callable: A preprocess function.
        """
        if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
            dataset['document'] = self.tokenize_progress(dataset, dataset_type, 'document')
            dataset['summary'] = self.tokenize_progress(dataset, dataset_type, 'summary')

            if dataset_type == 'train':
                self._vocab = Vocabulary.from_dataset(dataset, field_name=['document', 'summary'], max_size=max_size,
                                                      min_freq=min_freq,
                                                      padding=padding, unknown=unknown)

            if not self._buckets:
                pad1 = Pad(max_length=self._max_length, pad_val=self._vocab.padding_idx,
                           truncate=self._truncation_strategy)
                pad2 = Pad(max_length=self._max_pair_length, pad_val=self._vocab.padding_idx,
                           truncate=self._truncation_strategy)

                def token_to_idx(row):
                    data = {"input_ids": [self._vocab[i] for i in row["document"]],
                            "output_ids": [self._vocab[i] for i in row["summary"]]}
                    data["input_length"] = len(data["input_ids"])
                    data["output_length"] = len(data["output_ids"])
                    data["input_ids"] = pad1(data["input_ids"])
                    data["output_ids"] = pad2(data["output_ids"])
                    if "label" in row.keys():
                        data["label"] = row["label"]
                    return data
            else:
                pad = Pad(pad_val=self._vocab.padding_idx, buckets=self._buckets, truncate=self._truncation_strategy)

                def token_to_idx(row):
                    data = {"input_ids": [self._vocab[i] for i in row["document"]],
                            "output_ids": [self._vocab[i] for i in row["summary"]]}
                    data["input_length"] = len(data["input_ids"])
                    data["output_length"] = len(data["output_ids"])
                    data["input_ids"] = pad(data["input_ids"])
                    data["output_ids"] = pad(data["output_ids"])
                    if len(data["input_ids"]) > len(data["output_ids"]):
                        data["output_ids"] = Pad.padding(data["output_ids"], len(data["input_ids"]),
                                                         self._vocab.padding_idx)
                    else:
                        data["input_ids"] = Pad.padding(data["input_ids"], len(data["output_ids"]),
                                                        self._vocab.padding_idx)
                    data["padding_length"] = len(data["input_ids"])
                    return data
        else:
            self._pretrained_model_inputs = list(
                self._tokenizer("", return_length=not isinstance(self._buckets, List) and not isinstance(
                    self._max_length, int)).data.keys())

            if not self._buckets:
                def token_to_idx(row):
                    model_inputs = self._tokenizer(row["document"], truncation=self._truncation_strategy,
                                                   padding="max_length", max_length=self._max_length)
                    with self._tokenizer.as_target_tokenizer():
                        label = self._tokenizer(row["summary"], truncation=self._truncation_strategy,
                                                padding="max_length", max_length=self._max_pair_length)
                    model_inputs["labels"] = label["input_ids"]
                    return model_inputs
            else:
                def token_to_idx(row):
                    document_length = len(self._tokenizer.tokenize(row["document"], add_special_tokens=True))
                    summary_length = len(self._tokenizer.tokenize(row["summary"], add_special_tokens=True))
                    d_i = 0
                    for d_i in self._buckets:
                        if d_i >= document_length:
                            break
                    s_i = 0
                    for s_i in self._buckets:
                        if s_i >= summary_length:
                            break
                    i = d_i if d_i > s_i else s_i
                    model_inputs = self._tokenizer(row["document"], truncation=self._truncation_strategy,
                                                   padding="max_length", max_length=i)

                    with self._tokenizer.as_target_tokenizer():
                        label = self._tokenizer(row["summary"], truncation=self._truncation_strategy,
                                                padding="max_length", max_length=i)
                    model_inputs["labels"] = label["input_ids"]
                    model_inputs["padding_length"] = len(model_inputs["input_ids"])
                    return model_inputs
        return token_to_idx

    def _process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str,
                 unknown: str, dataset_type: str) -> DataFrame:
        # Whether using a pretrained model tokenizer.
        if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
            dataset["document"] = self.tokenize_progress(dataset, dataset_type, "document")
            dataset["summary"] = self.tokenize_progress(dataset, dataset_type, "summary")

            if dataset_type == 'train':
                self._vocab = Vocabulary.from_dataset(dataset, field_name=['document', 'summary'], max_size=max_size,
                                                      min_freq=min_freq,
                                                      padding=padding, unknown=unknown)
            dataset["input_ids"] = self._vocab.word_to_idx(dataset["document"])
            dataset["output_ids"] = self._vocab.word_to_idx(dataset["summary"])
            dataset.drop("document", axis=1, inplace=True)
            dataset.drop("summary", axis=1, inplace=True)
            dataset["input_length"] = self.get_length_progress(dataset, dataset_type, "input_ids")
            dataset["output_length"] = self.get_length_progress(dataset, dataset_type, "output_ids")
            if not self._buckets:
                if isinstance(self._max_length, int):
                    max_length1 = self._max_length
                else:
                    max_length1 = dataset["input_length"].max()
                if isinstance(self._max_pair_length, int):
                    max_length2 = self._max_pair_length
                else:
                    max_length2 = dataset["output_length"].max()
                pad1 = Pad(max_length=max_length1, pad_val=self._vocab.padding_idx, truncate=self._truncation_strategy)
                pad2 = Pad(max_length=max_length2, pad_val=self._vocab.padding_idx, truncate=self._truncation_strategy)
                dataset["input_ids"] = self.padding_progress(dataset, dataset_type, field="input_ids",
                                                             pad_function=pad1)
                dataset["output_ids"] = self.padding_progress(dataset, dataset_type, field="output_ids",
                                                              pad_function=pad2)
            else:
                pad = Pad(pad_val=self._vocab.padding_idx, buckets=self._buckets, truncate=self._truncation_strategy)
                dataset["input_ids"] = self.padding_progress(dataset, dataset_type, field="input_ids",
                                                             pad_function=pad)
                dataset["output_ids"] = self.padding_progress(dataset, dataset_type, field="output_ids",
                                                              pad_function=pad)
                dataset[["input_ids", "output_ids"]] = self.padding_same_progress(dataset, dataset_type,
                                                                                  ["input_ids", "output_ids"])
                dataset["padding_length"] = self.get_length_progress(dataset, dataset_type, "input_ids")
        else:
            self._pretrained_model_inputs_document = list(
                self._tokenizer("", return_length=not isinstance(self._buckets, List) and not isinstance(
                    self._max_length, int)).data.keys())
            self._pretrained_model_inputs_summary = list(
                self._tokenizer("", return_length=not isinstance(self._buckets, List) and not isinstance(
                    self._max_pair_length, int)).data.keys())
            document = DataFrame(self.tokenize_progress(dataset, dataset_type, field="document"))
            dataset.drop("document", axis=1, inplace=True)

            temp_pair_length = self._max_pair_length
            temp_length = self._max_length
            self._max_length = self._max_pair_length

            summary = DataFrame(self.tokenize_progress(dataset, dataset_type, field="summary"))
            dataset.drop("summary", axis=1, inplace=True)

            self._max_pair_length = temp_pair_length
            self._max_length = temp_length

            def document_list_split(row):
                data = row["document"]
                return tuple(data)

            document = document.apply(document_list_split, axis=1, result_type="expand")

            def summary_list_split(row):
                data = row["summary"]
                return tuple(data)

            summary = summary.apply(summary_list_split, axis=1, result_type="expand")
            if not isinstance(self._buckets, List) and not isinstance(self._max_length, int):
                document.columns = self._pretrained_model_inputs_document
                self._max_length = document["length"].max()
                document = DataFrame(
                    self.padding_progress(document, dataset_type, pad_function=self._tokenizer.pad))
                document.columns = self._pretrained_model_inputs_document
                document.drop("length", axis=1, inplace=True)
                self._pretrained_model_inputs_document.remove("length")
            else:
                document.columns = self._pretrained_model_inputs_document

            if not isinstance(self._buckets, List) and not isinstance(self._max_pair_length, int):
                summary.columns = self._pretrained_model_inputs_summary
                self._max_pair_length = summary["length"].max()
                temp_pair_length = self._max_pair_length
                temp_length = self._max_length
                self._max_length = self._max_pair_length
                summary = DataFrame(
                    self.padding_progress(summary, dataset_type, pad_function=self._tokenizer.pad))
                self._max_pair_length = temp_pair_length
                self._max_length = temp_length
                summary.columns = self._pretrained_model_inputs_summary
                summary.drop("length", axis=1, inplace=True)
                self._pretrained_model_inputs_summary.remove("length")
            else:
                summary.columns = self._pretrained_model_inputs_summary

            dataset[document.columns] = document
            dataset["labels"] = summary["input_ids"]
            del document
            del summary
            if isinstance(self._buckets, List):
                dataset["input_ids_length"] = self.get_length_progress(dataset, dataset_type, "input_ids")
                dataset["labels_length"] = self.get_length_progress(dataset, dataset_type, "labels")
                group = dataset.groupby("input_ids_length")
                for i in group:
                    _, dataset_group = i
                    self._max_length = dataset_group["labels_length"].max()
                    dataset_group = DataFrame(
                        self.padding_progress(DataFrame({"input_ids": dataset_group['labels']}), dataset_type,
                                              pad_function=self._tokenizer.pad))
                    dataset_group.columns = self._pretrained_model_inputs
                    dataset['labels'][dataset_group.index] = dataset_group['input_ids']
                dataset["padding_length"] = self.get_length_progress(dataset, dataset_type, "input_ids")
            self._pretrained_model_inputs = self._pretrained_model_inputs_document
        return dataset

    def load(self, paths: Optional[str] = None) -> Dict[str, DataFrame]:
        self._datasets = load_dataset(paths)
        self._datasets['dev'] = self._datasets.pop('validation')
        dict_dataset = {}
        for i in self._datasets.keys():
            pd_dataset = pd.DataFrame(columns=self._datasets[i].column_names)
            for j in self._datasets[i].column_names:
                pd_dataset[j] = pd.Series(self._datasets[i][j])
            dict_dataset[i] = pd_dataset
        self._datasets = dict_dataset
        return self._datasets

    def _write_to_mr(self, dataset: DataFrame, file_path: Union[str, Dict[int, str]], is_test: bool,
                     process_function: callable = None) -> List[str]:
        """
        Write CLSDataset to .mindrecord file.

        Args:
            dataset (DataFrame): Tokenizer function.
            file_path (Union[str, Dict[int, str]]): Path of mindrecord file.
            is_test (bool): Whether the data set is a test set.
            process_function (callable): A function is used to preprocess data.

        Returns:
            List[str]: Dataset field
        """
        if isinstance(file_path, Dict):
            writer = {}
            for k, v in file_path.items():
                writer[k] = FileWriter(file_name=v, shard_num=1)
        else:
            writer = FileWriter(file_name=file_path, shard_num=1)
        # Whether using a pretrained model tokenizer.
        if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
            data_schema = {
                'input_ids': {'type': 'int32', 'shape': [-1]},
                'input_length': {'type': 'int32', 'shape': [-1]}}
        else:
            data_schema = {}
            for i in self._pretrained_model_inputs:
                data_schema[i] = {'type': 'int32', 'shape': [-1]}

        if not is_test:
            if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
                data_schema["output_ids"] = {"type": "int32", "shape": [-1]}
                data_schema["output_length"] = {"type": "int32", "shape": [-1]}
            else:
                data_schema["labels"] = {"type": "int32", "shape": [-1]}

        if isinstance(writer, Dict):
            for k in file_path.keys():
                writer[k].add_schema(data_schema, self._name)
        else:
            writer.add_schema(data_schema, self._name)
        if not isinstance(writer, Dict):
            data = []
        vocab_bar = tqdm(dataset.iterrows(), total=len(dataset))
        for index, row in vocab_bar:
            # Whether using a pretrained model tokenizer.
            if callable(process_function):
                row = process_function(row)
            if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
                sample = {'input_ids': np.array(row["input_ids"], dtype=np.int64),
                          'input_length': np.array(row["input_length"], dtype=np.int64)}
            else:
                sample = {}
                for i in self._pretrained_model_inputs:
                    sample[i] = np.array(row[i], dtype=np.int64)

            if not is_test:
                if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
                    sample['output_ids'] = np.array(row["output_ids"], dtype=np.int64)
                    sample['output_length'] = np.array(row["output_length"], dtype=np.int64)
                else:
                    sample['labels'] = np.array(row['labels'], dtype=np.int64)
            if not isinstance(writer, Dict):
                data.append(sample)
                if index % 10 == 0:
                    writer.write_raw_data(data)
                    data = []
            else:
                if row["padding_length"] > list(writer.keys())[-1]:
                    writer[list(writer.keys())[-1]].write_raw_data([sample])
                else:
                    writer[row["padding_length"]].write_raw_data([sample])
            vocab_bar.set_description("Writing data to .mindrecord file")
        if not isinstance(writer, Dict):
            if data:
                writer.write_raw_data(data)
        if not isinstance(writer, Dict):
            writer.commit()
        else:
            for v in writer.values():
                v.commit()
        return list(data_schema.keys())

    def _load(self, path: str) -> DataFrame:
        pass
