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
"""Embedding utils."""
from typing import Optional
import numpy as np

import mindspore
import mindspore.nn as nn


def get_embeddings(init_embed, padding_idx: Optional[int] = None):
    """
    Returns the Embedding object based on the input init_embed. If the input is tuple, a nn.Embedding is randomly
    initialized. If the input is numpy.ndarray type, nn.Embedding is initialized with numpy.ndarray value;
    If mindspore.Tensor, initialize nn.Embedding with Tensor value; If the input is embedding object of mindspore,
    the original object is returned without processing.
    """
    if isinstance(init_embed, tuple):
        embeddings = nn.Embedding(vocab_size=init_embed[0], embedding_size=init_embed[1], padding_idx=padding_idx)
    elif isinstance(init_embed, nn.Cell):
        embeddings = init_embed
    elif isinstance(init_embed, mindspore.Tensor):
        embeddings = nn.Embedding(vocab_size=init_embed.shape[0], embedding_size=init_embed.shape[1],
                                  embedding_table=init_embed)
    elif isinstance(init_embed, np.ndarray):
        init_embed = mindspore.Tensor(init_embed, dtype=mindspore.float32)
        embeddings = nn.Embedding(vocab_size=init_embed.shape[0], embedding_size=init_embed.shape[1],
                                  embedding_table=init_embed)
    else:
        raise TypeError(f"invalid init_embed type: {type(init_embed)}")
    return embeddings
