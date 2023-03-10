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
"""Encoder basic model"""

import mindspore.nn as nn


class EncoderBase(nn.Cell):
    r"""
    Basic class for encoders

    Inputs:
        - **src_token** (Tensor) - Tokens in the source language with shape [batch, max_len].
        - **src_length** (Tensor) - Lengths of each sentence with shape [batch].
        - **mask** (Tensor) - Its elements identify whether the corresponding input token is padding or not.
            If True, not padding token. If False, padding token. Defaults to None.
    """

    def construct(self, src_token, src_length=None, mask=None):
        raise NotImplementedError("Model must implement the construct method")

    def reorder_encoder_out(self, encoder_out, new_order):
        """Reorder encoder output according to `new_order`."""
        raise NotImplementedError

    def reset_parameters(self, mask=None):
        """Reset model's parameters"""
        raise NotImplementedError
