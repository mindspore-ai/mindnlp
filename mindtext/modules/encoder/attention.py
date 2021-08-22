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
"""Attention classes for Encoder."""

import math
from typing import Optional

import mindspore
from mindspore.common import dtype as mstype
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer as init
from mindspore.common.initializer import XavierUniform as xavUniform
from mindspore import numpy as msnp

__all__ = [
    "DotAttention",
    "MultiHeadAttention",
    "SelfAttention",
]


class CastWrapper(nn.Cell):
    """
    Cast wrapper.
    """

    def __init__(self, src_type: mindspore.dtype = mstype.float32, dst_type: mindspore.dtype = mstype.float32):
        super(CastWrapper, self).__init__()
        self.cast = P.Cast()
        self.scr_type = src_type
        self.dst_type = dst_type

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        Args:
            x (Tensor): The shape of tensor is (x1,x2,...,xR). The tensor to be cast.

        Returns:
            Tensor, the shape of tensor is the same as x.
        """
        return self.cast(x, self.dst_type)


class LayerPreprocess(nn.Cell):
    """
    Preprocess input of each layer.

    Args:
        in_channels (int): The size of input channel, generally, last dim of input.

    """

    def __init__(self, in_channels: Optional[int] = None):
        super(LayerPreprocess, self).__init__()
        self.layernorm = nn.LayerNorm((in_channels,))
        self.cast = P.Cast()
        self.get_dtype = P.DType()

    def construct(self, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Args:
            input_tensor (Tensor): The input of preprocess layer.

        Returns:
            outputs (Tensor): The output of preprocess layer.
        """
        output = self.cast(input_tensor, mstype.float32)
        output = self.layernorm(output)
        output = self.cast(output, self.get_dtype(input_tensor))
        return output


class LayerPostprocess(nn.Cell):
    """
    Postprocess output of each layer.

    Args:
        dropout_prob (float): The dropout probability for postprocess layer. Default: 0.1.

    """

    def __init__(self,
                 dropout_prob: float = 0.1):
        super(LayerPostprocess, self).__init__()
        self.add = P.Add()
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.use_dropout = dropout_prob > 0

    def construct(self, hidden_tensor: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Args:
            hidden_tensor (Tensor): The output of hidden layer.
            input_tensor (Tensor): The input of hidden layer.

        Returns:
            output (Tensor): The output of postprocess layer.
        """
        output = hidden_tensor
        if self.use_dropout:
            output = self.dropout(output)
        output = self.add(output, input_tensor)
        return output


class DotAttention(nn.Cell):
    """
    DotAttention in Transformer.

    Args:
        key_size (int): The size of last dim of Key.
        value_size (int): The size of last dim of Value.
        dropout (int): The dropout rate of outputs. Default: 0.0.
    """

    def __init__(self, key_size: int, value_size: int, dropout: float = 0.0, has_attn_mask: bool = False):
        super(DotAttention, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.scale = math.sqrt(key_size)
        self.drop = nn.Dropout(keep_prob=1 - dropout)
        self.has_attn_mask = has_attn_mask
        self.softmax = nn.Softmax(axis=-1)
        self.matmul = nn.MatMul()
        self.select = P.Select()

    def construct(self, q: mindspore.Tensor, k: mindspore.Tensor, v: mindspore.Tensor,
                  attn_mask: Optional[mindspore.Tensor] = None) -> mindspore.Tensor:
        """
        Args:
            q (Tensor): The shape is (batch_size, q_len, q_size). The queries for Attention.
            k (Tensor):  The shape is (batch_size, k_len, k_size). The keys for Attention.
            v (Tensor): The shape is (batch_size, v_len, v_size). The values for Attention.
            attn_mask (Tensor): The is shape (batch_size, q_len, q_len). The mask matrix for Attention,
                                the values should be True or False. Default: None.

        Returns:
            output (Tensor): The output of DotAttention.
        """
        attn = self.matmul(q, k.transpose((0, 2, 1))) / self.scale
        if self.has_attn_mask:
            attn_mask = attn_mask.astype(mstype.bool_)
            mask_full = msnp.full_like(attn, -1e9)
            attn = self.select(attn_mask, mask_full, attn)
        attn = self.softmax(attn)
        attn = self.drop(attn)
        output = self.matmul(attn, v)
        return output


class MultiHeadAttention(nn.Cell):
    """
    Apply multi-headed attention from "from_tensor" to "to_tensor".

    Args:
        batch_size (int): Batch size of input datasets.
        from_tensor_width (int): Size of last dim of from_tensor.
        to_tensor_width (int): Size of last dim of to_tensor.
        out_tensor_width (int): Size of last dim of out_tensor.
        num_heads (int): Number of attention heads. Default: 1.
        query_act (str): Activation function for the query transform. Default: None.
        key_act (str): Activation function for the key transform. Default: None.
        value_act (str): Activation function for the value transform. Default: None.
        has_attn_mask (bool): Specifies whether to use attention mask. Default: False.
        attn_dropout_prob (float): The dropout probability for
                                      MultiHeadAttention. Default: 0.0.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        compute_type (:class:`mindspore.dtype`): Compute type in MultiHeadAttention. Default: mstype.float32.
    """

    def __init__(self,
                 batch_size: int,
                 from_tensor_width: int,
                 to_tensor_width: int,
                 out_tensor_width: int,
                 num_heads: int = 1,
                 query_act: Optional[str] = None,
                 key_act: Optional[str] = None,
                 value_act: Optional[str] = None,
                 out_act: Optional[str] = None,
                 has_attn_mask: bool = True,
                 has_key_padding_mask: bool = False,
                 attn_dropout_prob: float = 0.0,
                 use_one_hot_embeddings: bool = False,
                 compute_type: mindspore.dtype = mstype.float32):
        super(MultiHeadAttention, self).__init__()
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.size_per_head = from_tensor_width // num_heads
        self.has_attn_mask = has_attn_mask
        self.has_key_padding_mask = has_key_padding_mask
        assert has_attn_mask
        self.use_one_hot_embeddings = use_one_hot_embeddings

        self.scores_mul = mindspore.Tensor([1.0 / math.sqrt(float(self.size_per_head))], dtype=compute_type)
        units = num_heads * self.size_per_head
        self.query_project = nn.Dense(from_tensor_width,
                                      units,
                                      activation=query_act,
                                      has_bias=False,
                                      weight_init=init(xavUniform(), [units, to_tensor_width])).to_float(compute_type)
        self.key_project = nn.Dense(to_tensor_width,
                                    units,
                                    activation=key_act,
                                    has_bias=False,
                                    weight_init=init(xavUniform(), [units, to_tensor_width])).to_float(compute_type)
        self.value_project = nn.Dense(to_tensor_width,
                                      units,
                                      activation=value_act,
                                      has_bias=False,
                                      weight_init=init(xavUniform(), [units, to_tensor_width])).to_float(compute_type)
        self.out_project = nn.Dense(units,
                                    out_tensor_width,
                                    activation=out_act,
                                    has_bias=False,
                                    weight_init=init(xavUniform(), [units, to_tensor_width])).to_float(compute_type)

        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.multiply = P.Mul()
        self.trans_shape = (0, 2, 1, 3)
        self.trans_shape_relative = (2, 0, 1, 3)
        self.trans_shape_position = (1, 2, 0, 3)
        self.multiply_data = mindspore.Tensor([-10000.0,], dtype=compute_type)
        self.batch_num = batch_size * num_heads
        self.matmul = P.BatchMatMul()

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(1 - attn_dropout_prob)
        self.use_dropout = attn_dropout_prob > 0

        if self.has_attn_mask:
            self.get_dtype = P.DType()
            self.ones = P.Ones()
            self.tril = nn.Tril()
            self.tile = P.Tile()

        self.cast_compute_type = CastWrapper(dst_type=compute_type)
        self.softmax_cast = P.Cast()
        self.select = P.Select()

    def construct(self, from_tensor: mindspore.Tensor, to_tensor: mindspore.Tensor,
                  attn_mask: Optional[mindspore.Tensor] = None,
                  key_padding_mask: Optional[mindspore.Tensor] = None) -> mindspore.Tensor:
        """
        Apply multi-head attention.

        Args:
            from_tensor (Tensor): The shape is (batch_size, from_seq_len, dim). The from_tensor sequence, generally
                                it's query tensor(Q) for attention.
            to_tensor (Tensor): The shape is (batch_size, to_seq_len, dim). The to_tensor sequences, generally it's key
                                tensor(K) and value tensor(V) for attention, K = V.
            attn_mask (Tensor): The shape is (from_seq_len, to_seq_len) or (batch_size, from_seq_len, to_seq_len).
                                The mask matrix(2D or 3D) for attention, the values should be [0/1] or [True/False].
                                Default: None.
            key_padding_mask (Tensor): The shape is (batch_size, from_seq_len). Used to indicate which positions of
                                        from_tensor are padding, and these padding positions will be ignored during
                                        attention calculation. The values should be [0/1] or [True/False].
                                        Default: None.

        Returns:
            output (Tensor): The output of multi-head attention.
        """
        from_seq_len = from_tensor.shape[1]
        to_seq_len = to_tensor.shape[1]
        shape_from = (self.batch_size, from_seq_len, self.num_heads, self.size_per_head)
        shape_to = (self.batch_size, to_seq_len, self.num_heads, self.size_per_head)
        shape_return = (self.batch_size, from_seq_len, self.num_heads * self.size_per_head)

        q = self.query_project(from_tensor)
        k = self.key_project(to_tensor)
        v = self.value_project(to_tensor)

        q = q.reshape(shape_from)
        q = q.transpose(self.trans_shape)
        k = k.reshape(shape_to)
        k = k.transpose(self.trans_shape)

        attn_scores = self.matmul_trans_b(q, k)
        attn_scores = self.multiply(attn_scores, self.scores_mul)

        if self.has_attn_mask:
            attn_mask = attn_mask.astype(mstype.bool_)
            dims = len(attn_mask.shape)
            if dims == 2:  # (from_seq_len, to_seq_len)
                # (from_seq_len, to_seq_len) ->  (1, 1, from_seq_len, to_seq_len)
                attn_mask = attn_mask.reshape(1, 1, from_seq_len, to_seq_len)
                # (1, 1, from_seq_len, to_seq_len) ->  (batch_size, num_heads, from_seq_len, to_seq_len)
                attn_mask = self.tile(attn_mask, (self.batch_size, self.num_heads, 1, 1))
            elif dims == 3:  # (batch_size, from_seq_len, to_seq_len)
                attn_mask = attn_mask.reshape(self.batch_size, 1, from_seq_len, to_seq_len)
                # (batch_size, 1,  from_seq_len, to_seq_len) -> (batch_size, num_heads, from_seq_len, to_seq_len)
                attn_mask = self.tile(attn_mask, (1, self.num_heads, 1, 1))

            mask_full = msnp.full_like(attn_scores, -1e9)
            attn_scores = self.select(attn_mask, mask_full, attn_scores)

        if self.has_key_padding_mask:
            key_padding_mask = key_padding_mask.astype(mstype.bool_)
            # (batch_size, from_seq_len) -> (batch_size, 1, from_seq_len, 1)
            key_padding_mask = key_padding_mask.reshape(self.batch_size, 1, from_seq_len, 1)
            # (batch_size, 1, from_seq_len, 1) - > (batch_size, num_heads, from_seq_len, to_seq_len)
            key_padding_mask = self.tile(key_padding_mask, (1, self.num_heads, 1, to_seq_len))
            key_mask_full = msnp.full_like(attn_scores, -1e9)
            attn_scores = self.select(key_padding_mask, key_mask_full, attn_scores)

        attn_scores = self.softmax_cast(attn_scores, mstype.float32)
        attn = self.softmax(attn_scores)
        attn = self.softmax_cast(attn, self.get_dtype(k))
        if self.use_dropout:
            attn = self.dropout(attn)

        v = v.reshape(shape_to)
        v = v.transpose(self.trans_shape)
        output = self.matmul(attn, v)

        output = output.transpose(self.trans_shape)
        output = output.reshape(shape_return)
        output = self.out_project(output)
        return output


class SelfAttention(nn.Cell):
    """
    Apply self-attention.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of attention layers.
        num_heads (int): Number of attention heads. Default: 1.
        attn_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one_hot encoding form. Default: False.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        has_attn_mask (bool): Specifies whether has attention mask. Default: True.
        is_encdec_attn (bool): Specifies whether query sequence and memory sequence are different. Default: False.
        compute_type (:class:`mindspore.dtype`): Compute type in MultiheadAttention. Default: mstype.float32.
    """

    def __init__(self,
                 batch_size: int,
                 hidden_size: int,
                 num_heads: int = 1,
                 attn_dropout_prob: float = 0.1,
                 use_one_hot_embeddings: bool = False,
                 hidden_dropout_prob: float = 0.1,
                 has_attn_mask: bool = True,
                 has_key_padding_mask: bool = False,
                 is_encdec_attn: bool = False,
                 compute_type: mindspore.dtype = mstype.float32):
        super(SelfAttention, self).__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (hidden_size, num_heads))
        self.size_per_head = int(hidden_size / num_heads)
        self.is_encdec_attn = is_encdec_attn

        self.attention = MultiHeadAttention(
            batch_size=batch_size,
            from_tensor_width=hidden_size,
            to_tensor_width=hidden_size,
            out_tensor_width=hidden_size,
            num_heads=num_heads,
            attn_dropout_prob=attn_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            has_attn_mask=has_attn_mask,
            has_key_padding_mask=has_key_padding_mask,
            compute_type=compute_type)

        self.hidden_size = hidden_size

        self.preprocess = LayerPreprocess(in_channels=self.hidden_size)
        self.postprocess = LayerPostprocess(dropout_prob=hidden_dropout_prob)

    def construct(self, input_tensor: mindspore.Tensor, memory_tensor: mindspore.Tensor,
                  attn_mask: Optional[mindspore.Tensor] = None,
                  key_padding_mask: Optional[mindspore.Tensor] = None) -> mindspore.Tensor:
        """

        Args:
            input_tensor (Tensor): The shape is (batch_size, seq_len, hidden_units). The input_tensor sequence,
                                    generally it's query tensor(Q) for self-attention.
            memory_tensor (Tensor): The shape is (batch_size, seq_len, hidden_units). The memory_tensor sequence,
                                    generally it's key tensor(K) and value tensor(V) for self-attention, K = V.
            attn_mask (Tensor): The shape is (from_seq_len, to_seq_len) or (batch_size, from_seq_len, to_seq_len).
                                The mask matrix(2D or 3D) for attention, the values should be [0/1] or [True/False].
                                Default: None.
            key_padding_mask (Tensor): The shape is (batch_size, from_seq_len). Used to indicate which positions of
                                        from_tensor are padding, and these padding positions will be ignored during
                                        attention calculation. The values should be [0/1] or [True/False].
                                        Default: None.

        Returns:
            output (Tensor): The output of self-attention.
        """
        input_shape = input_tensor.shape
        input_tensor = input_tensor.reshape(-1, self.hidden_size)
        # layer norm
        output = self.preprocess(input_tensor)

        if not self.is_encdec_attn:
            memory_tensor = output

        output = output.reshape(input_shape)
        memory_tensor = memory_tensor.reshape(input_shape)
        attention_output = self.attention(output, memory_tensor, attn_mask, key_padding_mask)
        input_tensor = input_tensor.reshape(input_shape)
        # dropout and residual
        output = self.postprocess(attention_output, input_tensor)
        return output
