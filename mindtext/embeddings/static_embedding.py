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

r"""
.. todo::
    该文件中主要包含的是静态的Embedding，如预训练的word2vec, glove以及fasttext等。
    给定预训练embedding的名称或路径，根据vocab从embedding中抽取相应的数据。
    如果没有找到，则会随机初始化一个值
"""
