# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Sequence-to-vector basic model"""
# pylint: disable=abstract-method
# pylint: disable=arguments-differ
import mindspore.nn as nn
import mindspore.ops as ops

from .base import BaseModel


class Seq2vecModel(BaseModel):
    r"""
    Basic class for seq2vec models

    Args:
        encoder (EncoderBase): The encoder.
        head (nn.Cell): The module to process encoder output.
        dropout (float): The drop out rate, greater than 0 and less equal than 1.
            If None, not dropping out input units. Drfault: None.

    Inputs:
        - **src_tokens** (Tensor) - Tokens of source sentences with shape [batch, src_len].
        - **mask** (Tensor) - Its elements identify whether the corresponding input token is padding or not.
          If True, not padding token. If False, padding token. Defaults to None.

    Returns:
        - **result** (Tensor) - The result vector of seq2vec model with shape [batch, label_num].
    """

    def __init__(self, encoder, head, dropout: float = None):
        super().__init__()
        self.encoder = encoder
        self.head = head
        if dropout is None:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(1 - dropout)

    def construct(self, src_tokens, mask=None):
        """Construct method"""
        if mask is None:
            mask = self._gen_mask(src_tokens)

        context = self.get_context(src_tokens, mask)

        if self.dropout is not None:
            context = self.dropout(context)

        result = self.head(context)
        # TODO: Whether to add reduction
        return result

    def get_context(self, src_tokens, mask=None):
        """Get Context from encoder."""
        if mask is None:
            mask = self._gen_mask(src_tokens)
        return self.encoder(src_tokens, mask=mask)

    def _gen_mask(self, inputs):
        """Generate mask tensor"""
        return ops.ones_like(inputs)
