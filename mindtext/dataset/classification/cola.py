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
"""cola class"""
import pandas as pd


class Cola:
    """ Usage of Dataset CoLA """

    def __init__(self):
        """"""

    def load(self, path):
        """input path and return raw DataSet samples """
        with open(path, "r", encoding='utf-8') as reader:
            data = pd.read_csv(reader, sep='\t', header=None)
        data.columns = ['source', 'judgeLabel', 'originalLabel', 'sentence']
        dataframe = pd.DataFrame(data)
        dataframe.fillna('', inplace=True)
        print(dataframe, '\n')
        return dataframe

    def process(self):
        """dataset process"""
        return None


if __name__ == "__main__":
    test = Cola()
    df = test.load('train.tsv')
    print(df.values[:, :])
